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

package owl.ltl;

import java.util.BitSet;
import java.util.Objects;
import java.util.function.Predicate;
import owl.util.ImmutableObject;

public abstract class UnaryModalOperator extends ImmutableObject implements Formula {
  public final Formula operand;

  UnaryModalOperator(Formula operand) {
    this.operand = operand;
  }

  @Override
  public boolean allMatch(Predicate<Formula> predicate) {
    return predicate.test(this) && operand.allMatch(predicate);
  }

  @Override
  public boolean anyMatch(Predicate<Formula> predicate) {
    return predicate.test(this) || operand.anyMatch(predicate);
  }

  @Override
  protected boolean equals2(ImmutableObject o) {
    UnaryModalOperator that = (UnaryModalOperator) o;
    return Objects.equals(operand, that.operand);
  }

  public Formula getOperand() {
    return operand;
  }

  public abstract String getOperator();

  @Override
  public Formula temporalStep(BitSet valuation) {
    return this;
  }

  @Override
  public Formula temporalStepUnfold(BitSet valuation) {
    return unfold();
  }

  @Override
  public String toString() {
    return getOperator() + operand;
  }
}
