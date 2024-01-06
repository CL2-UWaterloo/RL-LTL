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

package owl.translations.frequency;

import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javax.annotation.Nonnull;
import jhoafparser.consumer.HOAConsumer;
import jhoafparser.consumer.HOAConsumerException;
import owl.automaton.output.HoaPrintable;
import owl.collections.ValuationSet;
import owl.factories.ValuationSetFactory;

@Deprecated
public class HoaConsumerGeneralisedRabin<S extends AutomatonState<?>> extends HoaConsumerExtended {
  private final GeneralizedRabinAcceptance2<S> acceptance;

  public HoaConsumerGeneralisedRabin(@Nonnull HOAConsumer hoa,
    ValuationSetFactory valuationSetFactory, List<String> aliases, Set<S> initialStates,
    @Nonnull GeneralizedRabinAcceptance2<S> accCond, int size) {
    super(hoa, valuationSetFactory.alphabetSize(), aliases, accCond, initialStates, size,
      EnumSet.allOf(HoaPrintable.HoaOption.class));
    this.acceptance = accCond;

    Map<String, List<Object>> map = acceptance.miscellaneousAnnotations();

    try {
      for (Entry<String, List<Object>> entry : map.entrySet()) {
        hoa.addMiscHeader(entry.getKey(), entry.getValue());
      }
    } catch (HOAConsumerException ex) {
      LOGGER.warning(ex.toString());
    }
  }

  @Override
  public void addEdge(ValuationSet key, AutomatonState<?> end) {
    Set<ValuationSet> realEdges = acceptance.getMaximallyMergedEdgesOfEdge(currentState, key);

    for (ValuationSet edgeKey : realEdges) {
      addEdgeBackend(edgeKey, end, acceptance.getInvolvedAcceptanceNumbers(currentState, edgeKey));
    }
  }
}
