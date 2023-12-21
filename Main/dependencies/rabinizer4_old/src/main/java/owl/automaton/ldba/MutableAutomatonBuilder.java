package owl.automaton.ldba;

import javax.annotation.Nullable;
import owl.automaton.MutableAutomaton;
import owl.automaton.acceptance.OmegaAcceptance;

public interface MutableAutomatonBuilder<S, T, U extends OmegaAcceptance> {
  @Nullable
  T add(S stateKey);

  MutableAutomaton<T, U> build();
}
