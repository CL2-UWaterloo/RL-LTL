package owl.translations.dra2dpa;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimaps;
import com.google.common.collect.Table;
import de.tum.in.naturals.IntPreOrder;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.BitSet;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import owl.automaton.Automaton;
import owl.automaton.AutomatonUtil;
import owl.automaton.MutableAutomaton;
import owl.automaton.MutableAutomatonFactory;
import owl.automaton.Views;
import owl.automaton.acceptance.GeneralizedRabinAcceptance.RabinPair;
import owl.automaton.acceptance.ParityAcceptance;
import owl.automaton.acceptance.ParityAcceptance.Parity;
import owl.automaton.acceptance.RabinAcceptance;
import owl.automaton.algorithms.SccDecomposition;
import owl.automaton.edge.Edge;
import owl.automaton.edge.LabelledEdge;
import owl.factories.ValuationSetFactory;

/**
 * Constructs the IAR parity automaton from the given Rabin automaton SCC.
 *
 * <p>The appearance record tracks the relative age of interesting events - in our case the
 * occurring Fin sets of the Rabin condition. Since there might be multiple events happening at
 * once, the ordering is not encoded as a permutation (of {@code {1,..,n}}) but rather as total
 * pre-orders of {@code {1,..n}}.</p>
 *
 * <p>The corresponding equivalence classes can be interpreted as "age classes" of events: An
 * element of the first class are of indistinguishable age, but all of them are younger than each
 * element of the next class and so on.</p>
 */
final class SccIARBuilder<R> {
  private static final Logger logger = Logger.getLogger(SccIARBuilder.class.getName());
  private final Table<R, IntPreOrder, IARState<R>> iarStates;
  private final RabinPair[] indexToPair;
  private final ImmutableSet<R> initialRabinStates;
  private final Automaton<R, RabinAcceptance> rabinAutomaton;
  private final MutableAutomaton<IARState<R>, ParityAcceptance> resultAutomaton;
  private final BitSet usedPriorities;

  private SccIARBuilder(Automaton<R, RabinAcceptance> rabinAutomaton,
    ImmutableSet<R> initialRabinStates, ImmutableSet<RabinPair> trackedPairs) {
    assert SccDecomposition.computeSccs(rabinAutomaton, initialRabinStates).size() == 1;
    assert rabinAutomaton.containsStates(initialRabinStates);

    this.rabinAutomaton = rabinAutomaton;
    this.initialRabinStates = initialRabinStates;
    this.usedPriorities = new BitSet(trackedPairs.size() * 2);
    this.usedPriorities.set(0);

    iarStates = HashBasedTable.create(rabinAutomaton.size(),
      rabinAutomaton.size() * trackedPairs.size());
    indexToPair = new RabinPair[trackedPairs.size()];

    int pairIndex = 0;
    for (RabinPair trackedPair : trackedPairs) {
      indexToPair[pairIndex] = trackedPair;
      pairIndex += 1;
    }

    ValuationSetFactory vsFactory = rabinAutomaton.getFactory();
    this.resultAutomaton = MutableAutomatonFactory.create(new ParityAcceptance(0, Parity.MIN_ODD),
      vsFactory);
  }

  static <R> SccIARBuilder<R> from(Automaton<R, RabinAcceptance> rabinAutomaton,
    Set<R> initialStates, Set<R> restriction, Set<RabinPair> trackedPairs) {
    return new SccIARBuilder<>(Views.filter(rabinAutomaton, restriction),
      ImmutableSet.copyOf(initialStates), ImmutableSet.copyOf(trackedPairs));
  }

  Automaton<IARState<R>, ParityAcceptance> build() {
    assert initialRabinStates.stream().allMatch(rabinAutomaton::containsState);
    logger.log(Level.FINE, "Building IAR automaton");

    Set<IARState<R>> initialStates = getInitialStates();
    resultAutomaton.setInitialStates(initialStates);
    logger.log(Level.FINEST, "Starting state space generation from {0}", initialStates);

    AutomatonUtil.exploreWithLabelledEdge(resultAutomaton, initialStates, state -> {
      // Compute the successors of the current state
      R rabinState = state.getOriginalState();
      IntPreOrder currentRecord = state.getRecord();

      // Iterate over each edge
      Set<LabelledEdge<IARState<R>>> successors = new HashSet<>();
      rabinAutomaton.forEachLabelledEdge(rabinState, (rabinEdge, valuations) -> {
        Edge<IARState<R>> iarEdge = computeSuccessorEdge(currentRecord, rabinEdge);
        successors.add(LabelledEdge.of(iarEdge, valuations));
      });
      return successors;
    });

    assert checkGeneratedStates();

    optimizeSuccessorRecord();
    optimizeInitialStates();

    int maximalUsedPriority = getMaximalUsedPriority();
    resultAutomaton.getAcceptance().setAcceptanceSets(maximalUsedPriority + 1);
    logger.log(Level.FINER, "Built automaton with {0} states and {1} priorities for input "
      + "automaton with {2} states and {3} pairs", new Object[] {resultAutomaton.size(),
      maximalUsedPriority, rabinAutomaton.size(), numberOfTrackedPairs()});
    return resultAutomaton;
  }

  private boolean checkGeneratedStates() {
    Set<R> stateSet = rabinAutomaton.getStates();
    Set<R> exploredRabinStates = resultAutomaton.getStates().stream()
      .map(IARState::getOriginalState)
      .collect(Collectors.toSet());
    assert stateSet.equals(exploredRabinStates) :
      String.format("Explored: %s%nExpected: %s", stateSet, exploredRabinStates);
    return true;
  }

  private Edge<IARState<R>> computeSuccessorEdge(IntPreOrder currentRecord, Edge<R> rabinEdge) {
    R rabinSuccessor = rabinEdge.getSuccessor();
    IntSet visitedFinSetIndices = new IntOpenHashSet();

    int classes = currentRecord.classes();
    int maximumPriority = classes * 2;
    int priority = maximumPriority;
    for (int currentClass = 0; currentClass < classes; currentClass++) {
      for (int rabinPairInClass : currentRecord.equivalenceClass(currentClass)) {
        RabinAcceptance.RabinPair rabinPair = indexToPair[rabinPairInClass];
        if (rabinEdge.inSet(rabinPair.finSet())) {
          visitedFinSetIndices.add(rabinPairInClass);
          priority = Math.min(priority, maximumPriority - 2 * currentClass - 2);
        } else if (rabinEdge.inSet(rabinPair.infSet())) {
          priority = Math.min(priority, maximumPriority - 2 * currentClass - 1);
        }
      }
    }

    usedPriorities.set(priority);

    IntPreOrder successorRecord = currentRecord.generation(visitedFinSetIndices);
    IARState<R> iarSuccessor = iarStates.row(rabinSuccessor)
      .computeIfAbsent(successorRecord, record -> IARState.active(rabinSuccessor, record));
    return Edge.of(iarSuccessor, priority);
  }

  private Set<IARState<R>> getInitialStates() {
    IntPreOrder initialRecord = IntPreOrder.coarsest(numberOfTrackedPairs());
    return initialRabinStates.stream()
      .map(initialRabinState -> IARState.active(initialRabinState, initialRecord))
      .collect(ImmutableSet.toImmutableSet());
  }

  private int getMaximalUsedPriority() {
    return usedPriorities.length() - 1;
  }

  private int numberOfTrackedPairs() {
    return indexToPair.length;
  }

  private void optimizeInitialStates() {
    /* Idea: Pick good initial permutations for the initial states and remove unreachable states */

    // Iterate in reverse topological order - the "best" SCCs should be further down the ordering
    List<Set<IARState<R>>> sccs = Lists.reverse(SccDecomposition.computeSccs(resultAutomaton));
    int rabinStateCount = rabinAutomaton.size();

    // We want to find the "optimal" SCC for each initial state. If we find a maximal SCC, we remove
    // the state from the search.
    Set<R> initialStatesToSearch = new HashSet<>(initialRabinStates);
    // Map each initial rabin state to the candidate IAR state
    Map<R, IARState<R>> initialStateCandidate = new HashMap<>(initialRabinStates.size());

    // Loop data structures:
    // Set of all rabin states in the current SCC
    Set<R> rabinStatesInScc = new HashSet<>();
    // Candidates for the initial states in the current SCC
    Map<R, IARState<R>> potentialInitialStates = new HashMap<>();

    for (Set<IARState<R>> scc : sccs) {
      if (SccDecomposition.isTransient(resultAutomaton::getSuccessors, scc)) {
        continue;
      }
      rabinStatesInScc.clear();
      potentialInitialStates.clear();

      // Gather all the rabin states in this SCC and see which initial states could be mapped here
      scc.forEach(sccState -> {
        R originalState = sccState.getOriginalState();
        rabinStatesInScc.add(originalState);
        if (initialStatesToSearch.contains(originalState)) {
          potentialInitialStates.put(originalState, sccState);
        }
      });

      if (potentialInitialStates.isEmpty()) {
        // This SCC does not contain any initial states - No point in continuing the search
        continue;
      }

      assert rabinStatesInScc.size() <= rabinStateCount;

      if (rabinStatesInScc.size() < rabinStateCount) {
        // Since we investigate an SCC in the Rabin automaton, we are guaranteed to find an SCC
        // containing all Rabin states. Hence, we can wait for the above case to occur
        continue;
      }

      // This definitely is the biggest SCC we can find (in terms of contained Rabin states)
      // -> We can stop the search for all contained initial states
      initialStateCandidate.putAll(potentialInitialStates);
      initialStatesToSearch.removeAll(potentialInitialStates.keySet());
      if (initialStatesToSearch.isEmpty()) {
        break;
      }
    }
    assert initialStateCandidate.size() == initialRabinStates.size();

    resultAutomaton.setInitialStates(initialStateCandidate.values());
    resultAutomaton.removeUnreachableStates();
  }

  private void optimizeSuccessorRecord() {
    /* Idea: The IAR records have a notion of "refinement". We now eliminate all states which are
     * refined by some other state. */

    // TODO: This could be done on-the-fly: While constructing the automaton, check for each
    // newly explored state if it refines / is refined by some state and act accordingly

    // In this table, we store for each pair (rabin state, record) the most refined state existing
    Table<R, IntPreOrder, IARState<R>> refinementTable = HashBasedTable.create();

    // Loop variables:
    // States for which we don't know where they are in the hierarchy
    Set<IARState<R>> unknownStates = new HashSet<>();
    // States which are not refined by any other state
    Set<IARState<R>> topElements = new HashSet<>();
    rabinAutomaton.forEachState(rabinState -> {
      assert topElements.isEmpty() && unknownStates.isEmpty();
      unknownStates.addAll(iarStates.row(rabinState).values());
      // Found refinements (mapping a particular record to its refining state)
      Map<IntPreOrder, IARState<R>> foundRefinements = refinementTable.row(rabinState);

      Iterator<IARState<R>> iterator = unknownStates.iterator();
      while (iterator.hasNext()) {
        IARState<R> currentState = iterator.next();
        IntPreOrder currentRecord = currentState.getRecord();
        iterator.remove();

        // First, see if this record is refined by a known top element
        Optional<IARState<R>> refiningTopStateOptional = topElements.parallelStream()
          .filter(state -> state.getRecord().refines(currentRecord)).findAny();
        if (refiningTopStateOptional.isPresent()) {
          // This state is refined by a top element - add it to the table and remove it from search
          // space
          foundRefinements.put(currentRecord, refiningTopStateOptional.get());
          continue;
        }

        // We did not find a existing refinement to attach to - search through all other records
        // Only search in the unrefined states - if this state would be refined by any state for
        // which a refinement was found already, we would have picked up that refinement in the
        // previous step
        Optional<IARState<R>> refiningUnknownStateOptional = unknownStates.parallelStream()
          .filter(state -> state.getRecord().refines(currentRecord)).findAny();

        if (refiningUnknownStateOptional.isPresent()) {
          IARState<R> refiningUnknownState = refiningUnknownStateOptional.get();
          foundRefinements.put(currentRecord, refiningUnknownState);
        } else {
          // This state is not refined by any other - it's a top element
          topElements.add(currentState);
        }
      }
      assert unknownStates.isEmpty();

      // All values of the refinement map should be top elements. But now, we might have
      // a -> b and b -> c in the table. Do a second pass to eliminate all such transitive entries.
      // Strategy: For each non-top value, search a top value which refines it (has to exist)

      Map<IntPreOrder, IARState<R>> refinementReplacement = new HashMap<>();

      foundRefinements.forEach((order, refiningState) -> {
        if (topElements.contains(refiningState)) {
          return;
        }
        IARState<R> refiningTopElement = topElements.parallelStream()
          .filter(topElement -> topElement.getRecord().refines(refiningState.getRecord()))
          .findAny().orElseThrow(AssertionError::new);
        refinementReplacement.put(order, refiningTopElement);
      });
      foundRefinements.putAll(refinementReplacement);

      topElements.clear();
    });

    // Now each refined state is mapped to a top refining state. Remap all edges to their refined
    // destination.

    if (refinementTable.isEmpty()) {
      logger.log(Level.FINE, "No refinements found");
    } else {
      if (logger.isLoggable(Level.FINEST)) {
        StringBuilder stringBuilder = new StringBuilder("Refinements:");
        rabinAutomaton.forEachState(rabinState -> {
          Map<IntPreOrder, IARState<R>> refinements = refinementTable.row(rabinState);
          if (refinements.isEmpty()) {
            return;
          }
          stringBuilder.append('\n').append(rabinState).append(':');
          Map<IARState<R>, Collection<IntPreOrder>> inverse =
            Multimaps.invertFrom(Multimaps.forMap(refinements), ArrayListMultimap.create()).asMap();
          inverse.forEach((iarState, stateRefinements) -> {
            stringBuilder.append("\n  ").append(iarState.getRecord()).append(" <-");
            stateRefinements.forEach(refinement -> stringBuilder.append(' ').append(refinement));
          });
        });
        logger.log(Level.FINEST, stringBuilder.toString());
        logger.log(Level.FINEST, "Automaton before refinement:\n{0}",
          AutomatonUtil.toHoa(resultAutomaton));
      }

      // Update initial states, for each initial state, pick its refinement (if there is any)
      resultAutomaton.setInitialStates(resultAutomaton.getInitialStates().stream()
        .map(initialState -> {
          IARState<R> refinedInitialState =
            refinementTable.get(initialState.getOriginalState(), initialState.getRecord());
          return Objects.requireNonNullElse(refinedInitialState, initialState);
        }).collect(ImmutableSet.toImmutableSet()));

      // Update edges
      resultAutomaton.remapEdges((state, edge) -> {
        // For each edge, pick the refined successor (if there is a refinement)
        IARState<R> successor = edge.getSuccessor();
        R rabinSuccessor = successor.getOriginalState();
        IARState<R> refinedSuccessor = refinementTable.get(rabinSuccessor, successor.getRecord());
        return refinedSuccessor == null ? edge : edge.withSuccessor(refinedSuccessor);
      });

      // Remove stale states
      resultAutomaton.removeStates(state ->
        refinementTable.contains(state.getOriginalState(), state.getRecord()));

      logger.log(Level.FINEST, () -> String.format("Automaton after refinement:%n%s",
        AutomatonUtil.toHoa(resultAutomaton)));
    }
  }
}
