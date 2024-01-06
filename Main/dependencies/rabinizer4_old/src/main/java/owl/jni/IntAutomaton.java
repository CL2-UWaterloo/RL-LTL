/*
 * Copyright (C) 2016  (See AUTHORS)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package owl.jni;

import static owl.translations.ltl2dpa.LTL2DPAFunction.RECOMMENDED_ASYMMETRIC_CONFIG;

import de.tum.in.naturals.bitset.BitSets;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import owl.automaton.Automaton;
import owl.automaton.AutomatonFactory;
import owl.automaton.AutomatonUtil;
import owl.automaton.MutableAutomaton;
import owl.automaton.acceptance.BuchiAcceptance;
import owl.automaton.acceptance.OmegaAcceptance;
import owl.automaton.acceptance.ParityAcceptance;
import owl.automaton.edge.Edge;
import owl.ltl.EquivalenceClass;
import owl.ltl.Formula;
import owl.ltl.Fragments;
import owl.ltl.LabelledFormula;
import owl.ltl.rewriter.RewriterFactory;
import owl.ltl.rewriter.ShiftRewriter;
import owl.ltl.rewriter.ShiftRewriter.ShiftedFormula;
import owl.run.DefaultEnvironment;
import owl.run.Environment;
import owl.translations.SimpleTranslations;
import owl.translations.ltl2dpa.LTL2DPAFunction;
import owl.translations.ltl2dpa.LTL2DPAFunction.Configuration;

// This is a JNI entry point. No touching.
@SuppressWarnings({"unused", "AssignmentOrReturnOfFieldWithMutableType"})
public final class IntAutomaton {
  private static final int NO_COLOUR = -1;
  private static final int NO_STATE = -1;

  private final Acceptance acceptance;
  private final int[] alphabetMapping;
  private final Automaton<Object, ?> automaton;
  private final List<Object> int2StateMap;
  private final Object2IntMap<Object> state2intMap;

  @SuppressWarnings("PMD.ArrayIsStoredDirectly")
  private IntAutomaton(Automaton<Object, ?> automaton, Acceptance acceptance, int[] mapping) {
    if (automaton.getInitialStates().isEmpty()) {
      this.automaton = AutomatonFactory.singleton(new Object(), automaton.getFactory(),
        BuchiAcceptance.INSTANCE);
    } else {
      this.automaton = AutomatonUtil.cast(automaton, Object.class, OmegaAcceptance.class);
    }

    int2StateMap = new ArrayList<>();
    int2StateMap.add(this.automaton.getInitialState());

    state2intMap = new Object2IntOpenHashMap<>();
    state2intMap.put(this.automaton.getInitialState(), 0);
    state2intMap.defaultReturnValue(NO_STATE);

    this.acceptance = acceptance;
    this.alphabetMapping = mapping;

    // Fix accepting sink to id 1.
    if (acceptance == Acceptance.CO_SAFETY) {
      EquivalenceClass trueClass = ((EquivalenceClass) this.automaton.getInitialState())
        .getFactory().getTrue();
      int index = lookup(trueClass);
      assert index == 1;
    }
  }

  private static Acceptance detectAcceptance(Automaton<?, ParityAcceptance> automaton) {
    ParityAcceptance parity = automaton.getAcceptance();

    switch (parity.getParity()) {
      case MAX_EVEN:
        return Acceptance.PARITY_MAX_EVEN;
      case MAX_ODD:
        return Acceptance.PARITY_MAX_ODD;
      case MIN_ODD:
        return Acceptance.PARITY_MIN_ODD;
      case MIN_EVEN:
        return Acceptance.PARITY_MIN_EVEN;
      default:
        throw new AssertionError();
    }
  }

  private static IntAutomaton of(Automaton<?, ?> automaton, Acceptance acceptance, int[] mapping) {
    return new IntAutomaton(AutomatonUtil.cast(automaton, Object.class, OmegaAcceptance.class),
      acceptance, mapping);
  }

  public static IntAutomaton of(Formula formula, boolean onTheFly) {
    Environment environment = DefaultEnvironment.standard();
    ShiftedFormula shiftedFormula = ShiftRewriter.shiftLiterals(RewriterFactory.apply(formula));
    LabelledFormula labelledFormula = Hacks.attachDummyAlphabet(shiftedFormula.formula);

    if (Fragments.isSafety(labelledFormula.formula)) {
      return of(SimpleTranslations.buildSafety(labelledFormula, environment),
        Acceptance.SAFETY, shiftedFormula.mapping);
    }
    if (Fragments.isCoSafety(labelledFormula.formula)) {
      return of(SimpleTranslations.buildCoSafety(labelledFormula, environment),
        Acceptance.CO_SAFETY, shiftedFormula.mapping);
    }
    if (Fragments.isDetBuchiRecognisable(labelledFormula.formula)) {
      return of(SimpleTranslations.buildBuchi(labelledFormula, environment),
        Acceptance.BUCHI, shiftedFormula.mapping);
    }
    if (Fragments.isDetCoBuchiRecognisable(labelledFormula.formula)) {
      return of(SimpleTranslations.buildCoBuchi(labelledFormula, environment),
        Acceptance.CO_BUCHI, shiftedFormula.mapping);
    }

    // Fallback to DPA
    Set<Configuration> configuration = EnumSet.copyOf(RECOMMENDED_ASYMMETRIC_CONFIG);

    if (onTheFly) {
      configuration.add(Configuration.GREEDY);
      configuration.add(Configuration.COLOUR_OVERAPPROXIMATION);
      configuration.remove(Configuration.OPTIMISE_INITIAL_STATE);
    }

    Automaton<?, ParityAcceptance> automaton =
      new LTL2DPAFunction(environment, configuration).apply(labelledFormula);
    assert !onTheFly || !(automaton instanceof MutableAutomaton)
      : "Internal Error: Automaton was explicitly constructed and not on-the-fly.";
    return of(automaton, detectAcceptance(automaton), shiftedFormula.mapping);
  }

  public int acceptance() {
    return acceptance.ordinal();
  }

  public int acceptanceSetCount() {
    return automaton.getAcceptance().getAcceptanceSets();
  }

  @SuppressWarnings("PMD.MethodReturnsInternalArray")
  public int[] alphabetMapping() {
    return alphabetMapping;
  }

  public int[] edges(int state) {
    Object o = int2StateMap.get(state);

    int i = 0;
    int size = automaton.getFactory().alphabetSize();
    int[] edges = new int[2 << size];

    for (BitSet valuation : BitSets.powerSet(size)) {
      Edge<?> edge = automaton.getEdge(o, valuation);

      if (edge == null) {
        edges[i] = NO_STATE;
        edges[i + 1] = NO_COLOUR;
      } else {
        edges[i] = lookup(edge.getSuccessor());
        edges[i + 1] = edge.largestAcceptanceSet();
      }

      i += 2;
    }

    return edges;
  }

  private int lookup(Object o) {
    int index = state2intMap.getInt(o);

    if (index == NO_STATE) {
      int2StateMap.add(o);
      state2intMap.put(o, int2StateMap.size() - 1);
      index = int2StateMap.size() - 1;
    }

    return index;
  }

  public int[] successors(int state) {
    Object o = int2StateMap.get(state);

    int i = 0;
    int size = automaton.getFactory().alphabetSize();
    int[] successors = new int[1 << size];

    for (BitSet valuation : BitSets.powerSet(size)) {
      Object successor = automaton.getSuccessor(o, valuation);
      successors[i] = successor == null ? NO_STATE : lookup(successor);
      i += 1;
    }

    return successors;
  }

  enum Acceptance {
    BUCHI, CO_BUCHI, CO_SAFETY, PARITY_MAX_EVEN, PARITY_MAX_ODD, PARITY_MIN_EVEN, PARITY_MIN_ODD,
    SAFETY
  }
}