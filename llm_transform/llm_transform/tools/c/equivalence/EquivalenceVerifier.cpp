#include "Passes.h"

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

namespace {
#define GEN_PASS_DEF_ARRAYDATAFLOWEQUIVALENCE
#include "Passes.h.inc"


/// Semantic equivalence for affine kernels at the stable-memref boundary.
///
/// This checker intentionally ignores local allocations introduced by
/// transformations. It only reasons about affine.load/affine.store operations
/// whose memref operand ultimately aliases a function BlockArgument.
class EquivalenceVerifier {
public:
  /// Main utility entry point. Returns true iff every original read/write or
  /// write/write ordering over stable memrefs is preserved by the transformed
  /// function.
  static bool verify(func::FuncOp original, func::FuncOp transformed,
                     raw_ostream *debugStream = nullptr) {
    EquivalenceVerifier verifier(debugStream);
    return verifier.run(original, transformed);
  }

private:
  enum class ScheduleKind { Constant, DomainVar };
  enum class RelationSide { Domain, Range };

  struct ScheduleComponent {
    static ScheduleComponent constant(int64_t value) {
      return {ScheduleKind::Constant, value, 0};
    }

    static ScheduleComponent domainVar(unsigned position) {
      return {ScheduleKind::DomainVar, 0, position};
    }

    ScheduleKind kind;
    int64_t value;
    unsigned domainPosition;
  };

  struct StableAccess {
    MemRefAccess access;
    Operation *op = nullptr;
    unsigned argNumber = 0;
    bool isStore = false;
    unsigned rank = 0;
    SmallVector<ScheduleComponent, 8> schedule;
    std::string equivalenceTag = "";
  };

  using AccessMap = DenseMap<unsigned, SmallVector<StableAccess, 8>>;

  explicit EquivalenceVerifier(raw_ostream *debugStream)
      : debugStream(debugStream) {}

  bool run(func::FuncOp original, func::FuncOp transformed) {
    log("=== begin equivalence check: @", original.getSymName(), " vs @",
        transformed.getSymName(), " ===");

    if (!sameSignature(original, transformed))
      return fail("function signatures differ");
    log("function signatures match");

    log("collecting stable memory accesses for original @",
        original.getSymName());
    AccessMap originalAccesses = collectStableMemoryAccesses(original);
    log("collecting stable memory accesses for transformed @",
        transformed.getSymName());
    AccessMap transformedAccesses = collectStableMemoryAccesses(transformed);

    log("original touches ", originalAccesses.size(),
        " stable memref arg(s); transformed touches ",
        transformedAccesses.size());

    for (auto &entry : originalAccesses) {
      unsigned argNumber = entry.first;
      ArrayRef<StableAccess> originalList = entry.second;
      auto transformedIt = transformedAccesses.find(argNumber);
      if (transformedIt == transformedAccesses.end())
        return fail("transformed function does not access stable memref arg ",
                    argNumber);

      ArrayRef<StableAccess> transformedList = transformedIt->second;
      log("--- checking stable memref arg ", argNumber, ": ",
          originalList.size(), " original access(es), ",
          transformedList.size(), " transformed access(es) ---");

      for (const StableAccess &producer : originalList) {
        for (const StableAccess &consumer : originalList) {
          if (!producer.isStore && !consumer.isStore) {
            log("skipping read/read pair on arg ", argNumber,
                " (no dependency to preserve)");
            continue;
          }

          if (!verifyOriginalPair(producer, consumer, transformedList))
            return false;
        }
      }
    }

    log("=== equivalence check passed ===");
    return true;
  }

  static bool sameSignature(func::FuncOp lhs, func::FuncOp rhs) {
    return lhs.getFunctionType() == rhs.getFunctionType();
  }

  bool verifyOriginalPair(const StableAccess &originalProducer,
                          const StableAccess &originalConsumer,
                          ArrayRef<StableAccess> transformedAccesses) {
    log("checking original dependency candidate:");
    log("  producer = ", accessToString(originalProducer));
    log("  consumer = ", accessToString(originalConsumer));

    FailureOr<IntegerRelation> originalOverlap =
        getSameElementRelation(originalProducer, originalConsumer);
    if (failed(originalOverlap))
      return fail("failed to build original access relation");
    logRelation("original same-element relation", *originalOverlap);

    // Original dependency relation:
    //   { producer_instance -> consumer_instance |
    //       access(producer_instance) == access(consumer_instance)
    //       and schedule(producer_instance) < schedule(consumer_instance) }
    if (!existsLexicographicOrdering(*originalOverlap,
                                     originalProducer.schedule,
                                     RelationSide::Domain,
                                     originalConsumer.schedule,
                                     RelationSide::Range,
                                     /*strict=*/true)) {
      log("  no original ordering producer < consumer; nothing to preserve");
      return true;
    }

    // A direction-free self-dependence — e.g. a reduction accumulator that is
    // loaded and stored through the *same* access function — orders the pair in
    // BOTH directions on the same memory element: over the original schedule,
    // producer < consumer holds for some instances and consumer < producer for
    // others (along the free reduction dimension, which the shared access
    // function collapses away). The element therefore pins no correspondence
    // between an original instance and its transformed counterpart on that
    // dimension, so there is no fixed producer/consumer order for this pair to
    // either preserve or reverse. Drawing a verdict from it produces a spurious
    // "reversal" (the reverse ordering already existed in the original), so
    // decline to conclude anything here. Genuine cross-iteration dependences
    // use distinct access functions, which pin a fixed inter-instance offset
    // and stay strictly one-directional, so they are still checked by the other
    // access pairs.
    if (existsLexicographicOrdering(*originalOverlap,
                                    originalConsumer.schedule,
                                    RelationSide::Range,
                                    originalProducer.schedule,
                                    RelationSide::Domain,
                                    /*strict=*/true)) {
      log("  original also orders consumer < producer on the same element "
          "(direction-free self-dependence, e.g. a reduction accumulator); no "
          "fixed order to preserve, skipping");
      return true;
    }

    log("  original dependency producer < consumer exists; searching for "
        "transformed counterpart");

    bool sawMappedPair = false;
    for (const StableAccess &transformedProducer : transformedAccesses) {
      if (!canBeCounterpart(originalProducer, transformedProducer))
        continue;

      for (const StableAccess &transformedConsumer : transformedAccesses) {
        if (!canBeCounterpart(originalConsumer, transformedConsumer))
          continue;

        log("  candidate transformed counterpart pair:");
        log("    producer = ", accessToString(transformedProducer));
        log("    consumer = ", accessToString(transformedConsumer));

        FailureOr<IntegerRelation> transformedOverlap =
            getSameElementRelation(transformedProducer, transformedConsumer);
        if (failed(transformedOverlap))
          return fail("failed to build transformed access relation");
        if (transformedOverlap->isEmpty()) {
          log("    transformed pair shares no memory element; skipping");
          continue;
        }
        logRelation("transformed same-element relation", *transformedOverlap);

        sawMappedPair = true;

        // Invalid transformed state:
        //   same memory element, but consumer is scheduled strictly before
        //   producer at a level that forces that reversal, or at the same
        //   timestamp for two distinct operations.
        //
        // The strict reversal must be *forced* at some schedule level rather
        // than merely realizable: a level that is order-free for this conflict
        // (a reduction index, free in the same-element relation) admits both
        // directions and reordering it preserves equivalence, so it must not be
        // mistaken for a dependency reversal.
        //
        // The "same operation" case skips the equal-time test to avoid
        // rejecting the trivial relation from an operation instance to itself.
        bool includeEqualTime =
            transformedProducer.op != transformedConsumer.op;
        if (existsForcedReversal(*transformedOverlap,
                                 transformedConsumer.schedule,
                                 RelationSide::Range,
                                 transformedProducer.schedule,
                                 RelationSide::Domain) ||
            (includeEqualTime &&
             existsLexicographicOrdering(*transformedOverlap,
                                         transformedConsumer.schedule,
                                         RelationSide::Range,
                                         transformedProducer.schedule,
                                         RelationSide::Domain,
                                         /*strict=*/false))) {
          return fail("transformed schedule reverses a stable-memref "
                      "dependency");
        }
        log("    transformed counterpart preserves the dependency order");
      }
    }

    if (!sawMappedPair)
      return fail("could not find transformed counterpart for stable-memref "
                  "dependency");

    log("  dependency preserved by all matched transformed counterparts");
    return true;
  }

  /// Builds `{ lhs_domain -> rhs_domain | lhs_access == rhs_access }` by
  /// composing the producer access relation with the inverse of the consumer
  /// access relation. `MemRefAccess::getAccessRelation` already contributes
  /// the affine loop-domain constraints and the affine index equalities.
  static FailureOr<IntegerRelation>
  getSameElementRelation(const StableAccess &lhs, const StableAccess &rhs) {
    IntegerRelation lhsAccess(PresburgerSpace::getRelationSpace());
    IntegerRelation rhsAccess(PresburgerSpace::getRelationSpace());
    if (failed(lhs.access.getAccessRelation(lhsAccess)) ||
        failed(rhs.access.getAccessRelation(rhsAccess)))
      return failure();

    rhsAccess.inverse();
    lhsAccess.compose(rhsAccess);
    return lhsAccess;
  }

  /// Counterpart matching is intentionally structural: stable argument number,
  /// load/store kind, and memref rank must match. Tiled/interchanged versions
  /// often rewrite loop IVs, so exact affine-map equality is too strong here.
  bool canBeCounterpart(const StableAccess &original,
                        const StableAccess &transformed) const {
    if (original.equivalenceTag.empty() && transformed.equivalenceTag.empty())
      // Emit warning if no equivalence tags are found, as this indicates a
      // failed mapping from original to transformed accesses. This could be due
      // to missing tags or a mismatch in the expected transformation pattern.
      emitWarning(original.op->getLoc())
                  << "Warning: no equivalence tags found for this access. "
                  << "This may indicate a failed mapping from original to "
                  << "transformed accesses. Consider adding equivalence tags "
                  << "to the relevant operations.\n";

    bool argMatch = original.argNumber == transformed.argNumber;
    bool tagMatch = original.equivalenceTag == transformed.equivalenceTag;
    bool kindMatch = original.isStore == transformed.isStore;
    bool rankMatch = original.rank == transformed.rank;
    bool match = argMatch && tagMatch && kindMatch && rankMatch;
    if (!match)
      log("    not a counterpart of [", accessToString(original),
          "]: arg=", argMatch, " tag=", tagMatch, " kind=", kindMatch,
          " rank=", rankMatch, " (candidate [", accessToString(transformed),
          "])");
    return match;
  }

  /// Checks whether `base` contains any point satisfying the lexicographic
  /// relation between two schedule vectors.
  ///
  /// `strict=true` constructs:
  ///   lhs < rhs
  ///
  /// `strict=false` constructs:
  ///   lhs == rhs
  ///
  /// Full lexicographic less-than is a disjunction:
  ///   lhs[0] < rhs[0]
  ///   OR lhs[0] == rhs[0] AND lhs[1] < rhs[1]
  ///   OR ...
  ///
  /// Each disjunct is tested with IntegerRelation::isEmpty(). This is the
  /// Presburger emptiness query that proves whether an invalid schedule state
  /// is realizable.
  bool existsLexicographicOrdering(
      const IntegerRelation &base, ArrayRef<ScheduleComponent> lhs,
      RelationSide lhsSide, ArrayRef<ScheduleComponent> rhs,
      RelationSide rhsSide, bool strict) const {
    unsigned common = std::min(lhs.size(), rhs.size());
    log("    testing ", (strict ? "strict (lhs < rhs)" : "equal (lhs == rhs)"),
        " ordering: lhs=", scheduleToString(lhs), sideSuffix(lhsSide),
        " rhs=", scheduleToString(rhs), sideSuffix(rhsSide));

    if (!strict) {
      if (lhs.size() != rhs.size()) {
        log("      schedule lengths differ (", lhs.size(), " vs ", rhs.size(),
            "); equal ordering impossible");
        return false;
      }
      IntegerRelation equalRel = base;
      for (unsigned i = 0; i < common; ++i)
        addEquality(equalRel, lhs[i], lhsSide, rhs[i], rhsSide);
      bool satisfiable = !equalRel.isEmpty();
      log("      equal-time relation is ",
          (satisfiable ? "satisfiable" : "empty"));
      return satisfiable;
    }

    for (unsigned differing = 0; differing < common; ++differing) {
      IntegerRelation disjunct = base;
      for (unsigned i = 0; i < differing; ++i)
        addEquality(disjunct, lhs[i], lhsSide, rhs[i], rhsSide);
      addStrictLess(disjunct, lhs[differing], lhsSide, rhs[differing],
                    rhsSide);
      std::string disjunctDesc =
          "      disjunct (equal prefix length " + std::to_string(differing) +
          ", strict-less at index " + std::to_string(differing) + ")";
      if (!disjunct.isEmpty()) {
        log(disjunctDesc, " is satisfiable -> ordering exists");
        return true;
      }
      log(disjunctDesc, " is empty");
    }

    // If all common components are equal, the shorter timestamp denotes the
    // surrounding static program point and executes before the longer one.
    if (lhs.size() < rhs.size()) {
      IntegerRelation prefixRel = base;
      for (unsigned i = 0; i < common; ++i)
        addEquality(prefixRel, lhs[i], lhsSide, rhs[i], rhsSide);
      bool satisfiable = !prefixRel.isEmpty();
      log("      lhs is a strict schedule prefix of rhs; prefix relation is ",
          (satisfiable ? "satisfiable -> ordering exists" : "empty"));
      return satisfiable;
    }

    log("      no satisfiable ordering disjunct found");
    return false;
  }

  /// Checks whether `base` *forces* the lexicographic ordering `lhs < rhs` at
  /// some schedule level: a level `k` where, with the higher-priority
  /// components held equal, `lhs < rhs` is realizable but `rhs < lhs` is NOT.
  ///
  /// This is the reversal-detection counterpart of `existsLexicographicOrdering`
  /// and is deliberately stricter. A bare existence query reports an ordering
  /// whenever *some* same-element instance pair realizes it — but when a
  /// schedule level is order-free for the conflict (both `lhs < rhs` and
  /// `rhs < lhs` are realizable with the prefix equal), reordering that level
  /// cannot reverse a genuine memory dependence: for every reversed instance
  /// pair there is a matching same-element pair in the original order. That is
  /// exactly what a reduction index produces — it is referenced by neither
  /// access function, so it stays free in the same-element relation, and tiling
  /// or interchanging it merely reshuffles accesses that already alias. Counting
  /// such an order-free level as a reversal is what spuriously rejected legal
  /// reduction schedules. A level is only a real reversal when the relation
  /// permits the reversed order there and forbids the forward one.
  bool existsForcedReversal(const IntegerRelation &base,
                            ArrayRef<ScheduleComponent> lhs,
                            RelationSide lhsSide,
                            ArrayRef<ScheduleComponent> rhs,
                            RelationSide rhsSide) const {
    unsigned common = std::min(lhs.size(), rhs.size());
    log("    testing forced reversal (lhs < rhs): lhs=", scheduleToString(lhs),
        sideSuffix(lhsSide), " rhs=", scheduleToString(rhs),
        sideSuffix(rhsSide));

    for (unsigned differing = 0; differing < common; ++differing) {
      IntegerRelation reversed = base;
      IntegerRelation forward = base;
      for (unsigned i = 0; i < differing; ++i) {
        addEquality(reversed, lhs[i], lhsSide, rhs[i], rhsSide);
        addEquality(forward, lhs[i], lhsSide, rhs[i], rhsSide);
      }
      addStrictLess(reversed, lhs[differing], lhsSide, rhs[differing], rhsSide);
      addStrictLess(forward, rhs[differing], rhsSide, lhs[differing], lhsSide);

      bool reversedSat = !reversed.isEmpty();
      bool forwardSat = !forward.isEmpty();
      std::string at =
          "      level " + std::to_string(differing) + " (prefix equal): ";
      if (reversedSat && !forwardSat) {
        log(at, "reversed order forced (forward order is impossible) -> "
                "genuine reversal");
        return true;
      }
      if (reversedSat && forwardSat) {
        log(at, "order-free for this conflict (both directions realizable); "
                "not a reversal, scanning deeper levels");
        continue;
      }
      log(at, "forward order holds (no reversal here)");
    }

    log("      no forced reversal at any constrained level");
    return false;
  }

  static void addEquality(IntegerRelation &rel, const ScheduleComponent &lhs,
                          RelationSide lhsSide,
                          const ScheduleComponent &rhs,
                          RelationSide rhsSide) {
    SmallVector<int64_t, 16> eq(rel.getNumCols(), 0);
    addExpression(eq, rel, lhs, lhsSide, /*scale=*/1);
    addExpression(eq, rel, rhs, rhsSide, /*scale=*/-1);
    rel.addEquality(eq);
  }

  static void addStrictLess(IntegerRelation &rel,
                            const ScheduleComponent &lhs,
                            RelationSide lhsSide,
                            const ScheduleComponent &rhs,
                            RelationSide rhsSide) {
    // rhs - lhs - 1 >= 0  <=>  lhs < rhs over integer schedules.
    SmallVector<int64_t, 16> ineq(rel.getNumCols(), 0);
    addExpression(ineq, rel, rhs, rhsSide, /*scale=*/1);
    addExpression(ineq, rel, lhs, lhsSide, /*scale=*/-1);
    ineq.back() -= 1;
    rel.addInequality(ineq);
  }

  static void addExpression(SmallVectorImpl<int64_t> &row,
                            const IntegerRelation &rel,
                            const ScheduleComponent &component,
                            RelationSide side,
                            int64_t scale) {
    if (component.kind == ScheduleKind::Constant) {
      row.back() += scale * component.value;
      return;
    }
    unsigned column = component.domainPosition;
    if (side == RelationSide::Range)
      column += rel.getNumDomainVars();
    row[column] += scale;
  }

  /// Builds a map from every memref Value that aliases, or holds a copy of, a
  /// function argument to that argument's number.
  ///
  /// Rather than walking up from a loaded memref to its root, this propagates
  /// *forward* from the function arguments: starting at each memref argument,
  /// it follows ViewLikeOpInterface ops (subview/cast/...) and CopyOpInterface
  /// ops (memref.copy) as undirected edges between buffers. This captures the
  /// staging patterns transformations introduce in both directions:
  ///
  ///   * packing/padding copies the argument *into* a local buffer
  ///     (`memref.copy %arg0, %subview` where %subview views a fresh alloc), so
  ///     the loads later issued against that alloc must resolve to %arg0;
  ///   * output write-back computes into a local alloc and copies it *into* the
  ///     result argument (`memref.copy %alloc, %arg2`), so the stores against
  ///     the alloc must resolve to %arg2.
  ///
  /// Because a copy makes its source and target hold equal data, both
  /// endpoints are treated as equivalent; view ops alias the same storage, so
  /// source and result are equivalent too. Any affine access on a buffer in an
  /// argument's set is attributed to that argument.
  DenseMap<Value, unsigned> buildArgEquivalenceMap(func::FuncOp func) const {
    DenseMap<Value, unsigned> argOf;

    for (BlockArgument arg : func.getArguments()) {
      if (!isa<MemRefType>(arg.getType()))
        continue;
      unsigned argNumber = arg.getArgNumber();

      SmallVector<Value, 8> worklist{arg};
      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        auto it = argOf.find(v);
        if (it != argOf.end()) {
          if (it->second != argNumber)
            log("  buffer is reachable from multiple arguments (", it->second,
                " and ", argNumber, "); keeping the first");
          continue;
        }
        argOf[v] = argNumber;
        log("  buffer ", v, " is equivalent to argument ", argNumber);

        // A view-like producer aliases the storage it views.
        if (auto viewOp = v.getDefiningOp<ViewLikeOpInterface>())
          worklist.push_back(viewOp.getViewSource());

        // Consumers: view results alias `v`, and either endpoint of a copy
        // holds the same data as the other.
        for (Operation *user : v.getUsers()) {
          if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
            if (viewOp.getViewSource() == v)
              for (Value result : viewOp->getResults())
                if (isa<MemRefType>(result.getType()))
                  worklist.push_back(result);
          } else if (auto copyOp = dyn_cast<CopyOpInterface>(user)) {
            worklist.push_back(copyOp.getSource());
            worklist.push_back(copyOp.getTarget());
          }
        }
      }
    }
    return argOf;
  }

  static std::optional<StringRef> getEquivalenceTag(Location loc) {
    if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
      StringRef name = nameLoc.getName().getValue();
      if (name.starts_with("eq_id_"))
        return name;
      // If the NameLoc wraps another location, keep digging
      return getEquivalenceTag(nameLoc.getChildLoc());
    }

    if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
      // If transformations fused multiple ops, check the underlying locations
      for (auto childLoc : fusedLoc.getLocations()) {
        if (auto tag = getEquivalenceTag(childLoc))
          return tag;
      }
    }

    // OpaqueLoc or CallSiteLoc could also be checked here if your
    // pipeline generates them, but NameLoc/FusedLoc is standard for Linalg.
    return std::nullopt;
  }

  AccessMap collectStableMemoryAccesses(func::FuncOp func) const {
    AccessMap accessMap;
    DenseMap<Value, unsigned> argOf = buildArgEquivalenceMap(func);

    func.walk([&](Operation *op) {
      Value memref;
      if (auto readOp = dyn_cast<AffineReadOpInterface>(op))
        memref = readOp.getMemRef();
      else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op))
        memref = writeOp.getMemRef();
      else
        return;

      auto it = argOf.find(memref);
      if (it == argOf.end()) {
        log("  ignoring ", op->getName().getStringRef(),
            ": memref is not equivalent to any function argument");
        return;
      }
      unsigned argNumber = it->second;

      std::string eqTag = "";
      if (std::optional<StringRef> tag = getEquivalenceTag(op->getLoc())) {
        eqTag = tag->str();
      }

      MemRefAccess access(op);
      StableAccess stableAccess{access,
                                op,
                                argNumber,
                                access.isStore(),
                                access.getRank(),
                                buildSchedule(op),
                                eqTag};
      log("  found ", accessToString(stableAccess));
      accessMap[stableAccess.argNumber].push_back(stableAccess);
    });

    return accessMap;
  }

  static SmallVector<ScheduleComponent, 8> buildSchedule(Operation *op) {
    SmallVector<Operation *, 8> path;
    for (
      Operation *cursor = op;
      cursor && !isa<func::FuncOp>(cursor);
      cursor = cursor->getParentOp()
    ) {
      path.push_back(cursor);
    }
    std::reverse(path.begin(), path.end());

    SmallVector<ScheduleComponent, 8> schedule;
    unsigned domainPosition = 0;
    for (Operation *pathOp : path) {
      schedule.push_back(ScheduleComponent::constant(getSiblingOrdinal(pathOp)));
      if (isa<AffineForOp>(pathOp))
        schedule.push_back(ScheduleComponent::domainVar(domainPosition++));
    }

    return schedule;
  }

  static int64_t getSiblingOrdinal(Operation *op) {
    int64_t ordinal = 0;
    for (Operation &sibling : *op->getBlock()) {
      if (&sibling == op)
        return ordinal;
      if (!sibling.hasTrait<OpTrait::IsTerminator>())
        ++ordinal;
    }
    return ordinal;
  }

  /// Streams a single log argument. `Value` (and its subclasses) are printed in
  /// their compact operand form (e.g. `%3`) rather than their full definition.
  template <typename T>
  static void streamArg(raw_ostream &os, T &&arg) {
    if constexpr (std::is_convertible_v<std::decay_t<T>, Value>) {
      Value(std::forward<T>(arg)).printAsOperand(os, OpPrintingFlags());
    } else {
      os << std::forward<T>(arg);
    }
  }

  template <typename... Args>
  bool fail(Args &&...args) const {
    if (debugStream) {
      (*debugStream) << "[array-dataflow-equivalence] ";
      (streamArg(*debugStream, std::forward<Args>(args)), ...);
      (*debugStream) << "\n";
    }
    return false;
  }

  /// Verbose trace line. No-op unless `verbose` enabled the debug stream.
  template <typename... Args>
  void log(Args &&...args) const {
    if (!debugStream)
      return;
    (*debugStream) << "[array-dataflow-equivalence] ";
    (streamArg(*debugStream, std::forward<Args>(args)), ...);
    (*debugStream) << "\n";
  }

  /// Dumps an IntegerRelation (dimensions plus body) under `label`. No-op
  /// unless verbose.
  void logRelation(StringRef label, const IntegerRelation &rel) const {
    if (!debugStream)
      return;
    (*debugStream) << "[array-dataflow-equivalence]   " << label
                   << " (domain=" << rel.getNumDomainVars()
                   << ", range=" << rel.getNumRangeVars()
                   << ", local=" << rel.getNumLocalVars() << "):\n";
    rel.print(*debugStream);
  }

  static StringRef sideSuffix(RelationSide side) {
    return side == RelationSide::Domain ? "(domain)" : "(range)";
  }

  static std::string scheduleToString(ArrayRef<ScheduleComponent> schedule) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << "[";
    for (unsigned i = 0; i < schedule.size(); ++i) {
      if (i)
        os << ", ";
      const ScheduleComponent &component = schedule[i];
      if (component.kind == ScheduleKind::Constant)
        os << "const(" << component.value << ")";
      else
        os << "iv(d" << component.domainPosition << ")";
    }
    os << "]";
    return os.str();
  }

  static std::string accessToString(const StableAccess &access) {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << (access.isStore ? "store" : "load") << " arg" << access.argNumber
       << " rank=" << access.rank << " tag='" << access.equivalenceTag
       << "' op='" << access.op->getName().getStringRef()
       << "' schedule=" << scheduleToString(access.schedule);
    return os.str();
  }

  raw_ostream *debugStream = nullptr;
};

struct ArrayDataflowEquivalencePass
    : public impl::ArrayDataflowEquivalenceBase<ArrayDataflowEquivalencePass> {
  using Base = impl::ArrayDataflowEquivalenceBase<ArrayDataflowEquivalencePass>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    func::FuncOp original = module.lookupSymbol<func::FuncOp>(originalFuncName);
    func::FuncOp transformed =
        module.lookupSymbol<func::FuncOp>(transformedFuncName);

    if (!original || !transformed) {
      SmallVector<func::FuncOp, 2> funcs;
      module.walk([&](func::FuncOp func) { funcs.push_back(func); });
      if (funcs.size() == 2) {
        original = funcs[0];
        transformed = funcs[1];
      }
    }

    if (!original) {
      module.emitError() << "could not find original function '"
                         << originalFuncName << "'";
      signalPassFailure();
      return;
    }
    if (!transformed) {
      module.emitError() << "could not find transformed function '"
                         << transformedFuncName << "'";
      signalPassFailure();
      return;
    }

    // Emit the verbose trace on stderr (unbuffered), not stdout: stdout must
    // carry only the verdict, and llvm::outs() is buffered so its trace can
    // flush past the wrapper's fd capture and corrupt the verdict line.
    raw_ostream *debug = verbose ? &llvm::errs() : nullptr;
    if (!EquivalenceVerifier::verify(original, transformed, debug)) {
      module.emitError() << "array dataflow equivalence check failed between @"
                         << original.getSymName() << " and @"
                         << transformed.getSymName();
      signalPassFailure();
    }
  }
};

static PassRegistration<ArrayDataflowEquivalencePass> passRegistration;

} // namespace

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ArrayDataflowEquivalence", "0.1",
          []() {}};
}
