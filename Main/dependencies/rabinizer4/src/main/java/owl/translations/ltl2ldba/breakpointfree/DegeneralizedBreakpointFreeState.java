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

package owl.translations.ltl2ldba.breakpointfree;

import java.util.Objects;
import javax.annotation.Nonnegative;
import javax.annotation.Nullable;
import owl.ltl.EquivalenceClass;
import owl.util.ImmutableObject;
import owl.util.StringUtil;

public final class DegeneralizedBreakpointFreeState extends ImmutableObject {

  @Nonnegative // Index of the current checked liveness (F) obligation.
  final int index;

  @Nullable
  final EquivalenceClass liveness;
  @Nullable
  final FGObligations obligations;
  @Nullable
  final EquivalenceClass safety;

  DegeneralizedBreakpointFreeState(@Nonnegative int index, @Nullable EquivalenceClass safety,
    @Nullable EquivalenceClass liveness, @Nullable FGObligations obligations) {
    assert 0 == index || (0 < index && index < obligations.liveness.length);

    this.index = index;
    this.liveness = liveness;
    this.obligations = obligations;
    this.safety = safety;
  }

  public static DegeneralizedBreakpointFreeState createSink() {
    return new DegeneralizedBreakpointFreeState(0, null, null, null);
  }

  @Override
  protected boolean equals2(ImmutableObject o) {
    DegeneralizedBreakpointFreeState that = (DegeneralizedBreakpointFreeState) o;
    return index == that.index && Objects.equals(safety, that.safety) && Objects
      .equals(liveness, that.liveness) && Objects.equals(obligations, that.obligations);
  }

  public FGObligations getObligations() {
    return obligations;
  }

  @Override
  protected int hashCodeOnce() {
    return Objects.hash(liveness, obligations, safety, index);
  }

  @Override
  public String toString() {
    return obligations + StringUtil.join(safety == null || safety.isTrue() ? null : "GWR=" + safety,
      liveness == null || liveness.isTrue() ? null : "FUM=" + liveness + " (" + index + ')');
  }
}