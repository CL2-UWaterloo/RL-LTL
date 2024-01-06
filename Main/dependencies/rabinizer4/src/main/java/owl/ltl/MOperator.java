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
import owl.ltl.visitors.BinaryVisitor;
import owl.ltl.visitors.IntVisitor;
import owl.ltl.visitors.Visitor;

/**
 * Strong Release.
 */
public final class MOperator extends BinaryModalOperator {

  public MOperator(Formula left, Formula right) {
    super(left, right);
  }

  public static Formula of(Formula left, Formula right) {
    if (left == BooleanConstant.FALSE || right == BooleanConstant.FALSE) {
      return BooleanConstant.FALSE;
    }

    if (left == BooleanConstant.TRUE) {
      return right;
    }

    if (left.equals(right)) {
      return left;
    }

    if (right == BooleanConstant.TRUE) {
      return FOperator.of(left);
    }

    return new MOperator(left, right);
  }

  @Override
  public int accept(IntVisitor v) {
    return v.visit(this);
  }

  @Override
  public <R> R accept(Visitor<R> v) {
    return v.visit(this);
  }

  @Override
  public <A, B> A accept(BinaryVisitor<B, A> v, B parameter) {
    return v.visit(this, parameter);
  }

  @Override
  public char getOperator() {
    return 'M';
  }

  @Override
  protected int hashCodeOnce() {
    return Objects.hash(MOperator.class, left, right);
  }

  @Override
  public boolean isPureEventual() {
    return false;
  }

  @Override
  public boolean isPureUniversal() {
    return false;
  }

  @Override
  public boolean isSuspendable() {
    return false;
  }

  @Override
  public WOperator not() {
    return new WOperator(left.not(), right.not());
  }

  @Override
  public Formula unfold() {
    return new Conjunction(right.unfold(), new Disjunction(left.unfold(), this));
  }

  @Override
  public Formula unfoldTemporalStep(BitSet valuation) {
    return Conjunction.of(right.unfoldTemporalStep(valuation),
      Disjunction.of(left.unfoldTemporalStep(valuation), this));
  }

}