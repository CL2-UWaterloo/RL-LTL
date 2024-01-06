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
 * Strong Until.
 */
public final class UOperator extends BinaryModalOperator {

  public UOperator(Formula left, Formula right) {
    super(left, right);
  }

  public static Formula of(Formula left, Formula right) {
    if (left == BooleanConstant.FALSE || right instanceof BooleanConstant) {
      return right;
    }

    if (left.equals(right)) {
      return left;
    }

    if (left == BooleanConstant.TRUE) {
      return FOperator.of(right);
    }

    return new UOperator(left, right);
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
    return 'U';
  }

  @Override
  protected int hashCodeOnce() {
    return Objects.hash(UOperator.class, left, right);
  }

  @Override
  public boolean isPureEventual() {
    return right.isPureEventual();
  }

  @Override
  public boolean isPureUniversal() {
    return left.isPureUniversal() && right.isPureUniversal();
  }

  @Override
  public boolean isSuspendable() {
    return right.isSuspendable();
  }

  @Override
  public ROperator not() {
    return new ROperator(left.not(), right.not());
  }

  @Override
  public Formula unfold() {
    return new Disjunction(right.unfold(), new Conjunction(left.unfold(), this));
  }

  @Override
  public Formula unfoldTemporalStep(BitSet valuation) {
    return Disjunction.of(right.unfoldTemporalStep(valuation),
      Conjunction.of(left.unfoldTemporalStep(valuation), this));
  }

}