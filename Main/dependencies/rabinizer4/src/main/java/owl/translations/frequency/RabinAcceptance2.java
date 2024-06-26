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

import java.util.Collections;
import java.util.List;

@Deprecated
public class RabinAcceptance2<S extends AutomatonState<?>> extends GeneralizedRabinAcceptance2<S> {
  public RabinAcceptance2(List<RabinPair2<TranSet<S>, List<TranSet<S>>>> acceptanceCondition) {
    super(acceptanceCondition);
    checkIfValidArgument(acceptanceCondition);
  }

  private void checkIfValidArgument(
    List<RabinPair2<TranSet<S>, List<TranSet<S>>>> acceptanceCondition) {
    for (RabinPair2<TranSet<S>, List<TranSet<S>>> pair : acceptanceCondition) {
      if (pair.right.size() != 1) {
        throw new IllegalArgumentException("Acceptance condition is not a Rabin condition.");
      }
    }

  }

  @Override
  public String getName() {
    return "Rabin";
  }

  @Override
  public List<Object> getNameExtra() {
    return Collections.singletonList(this.acceptanceCondition.size());
  }
}
