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

package owl.translations.nba2dpa;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThanOrEqualTo;

import jhoafparser.consumer.HOAConsumerNull;
import jhoafparser.consumer.HOAIntermediateCheckValidity;
import jhoafparser.parser.generated.ParseException;
import org.junit.Test;
import owl.automaton.Automaton;
import owl.automaton.AutomatonReader;
import owl.automaton.AutomatonReader.HoaState;
import owl.automaton.acceptance.GeneralizedBuchiAcceptance;
import owl.automaton.acceptance.ParityAcceptance;
import owl.run.DefaultEnvironment;

public class NBA2DPATest {

  private static final String INPUT = "HOA: v1\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "AP: 1 \"a\"\n"
    + "--BODY--\n"
    + "State: 0 {0}\n"
    + " [0]   1 \n"
    + "State: 1 \n"
    + " [t]   0 \n"
    + " [!0]  1 \n"
    + "--END--";

  private static final String INPUT2 = "HOA: v1\n"
    + "States: 1\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "AP: 1 \"a\"\n"
    + "--BODY--\n"
    + "State: 0\n"
    + "[t] 0 {0}\n"
    + "--END--";

  private static final String INPUT3 = "HOA: v1\n"
    + "States: 3\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label\n"
    + "AP: 2 \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 1\n"
    + "[0] 1 {0}\n"
    + "[!0] 2 {0}\n"
    + "State: 0\n"
    + "[0] 1\n"
    + "[t] 0\n"
    + "[!1] 2\n"
    + "State: 2\n"
    + "[0 & 1] 1 {0}\n"
    + "[!0 & 1] 2 {0}\n"
    + "--END--";

  private static final String INPUT4 = "HOA: v1\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label\n"
    + "AP: 1 \"a\"\n"
    + "--BODY--\n"
    + "State: 0\n"
    + "[!0] 0\n"
    + "[0] 1\n"
    + "State: 1\n"
    + "[t] 1 {0}\n"
    + "--END--";

  private static final String INPUT5 = "HOA: v1\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 1 \"a\"\n"
    + "--BODY--\n"
    + "State: 1 \"0\"\n"
    + "[0] 1 {0}\n"
    + "State: 0 \"1\"\n"
    + "[0] 1\n"
    + "[t] 0\n"
    + "--END--";

  private static final String INPUT6 = "HOA: v1\n"
    + "States: 1\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 1 \"a\"\n"
    + "--BODY--\n"
    + "State: 0 \n"
    + "[0] 0\n"
    + "[!0] 0 {0}\n"
    + "--END--";

  private static final String INPUT7 = "HOA: v1\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 2 \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 0 \n"
    + "[t] 1\n"
    + "State: 1\n"
    + "[!1] 1 {0}\n"
    + "[!0 & 1] 1"
    + "--END--";

  private static final String INPUT8 = "HOA: v1\n"
    + "States: 1\n"
    + "Start: 0\n"
    + "acc-name: generalized-Buchi 3\n"
    + "Acceptance: 3 Inf(0) & Inf(1) & Inf(2)\n"
    + "properties: trans-acc trans-label \n"
    + "properties: deterministic \n"
    + "AP: 3 \"a\" \"b\" \"c\"\n"
    + "--BODY--\n"
    + "State: 0\n"
    + "[0 & 1 & 2] 0\n"
    + "[!0 & 1 & 2] 0 {0}\n"
    + "[0 & !1 & 2] 0 {1}\n"
    + "[!0 & !1 & 2] 0 {0 1}\n"
    + "[0 & 1 & !2] 0 {2}\n"
    + "[!0 & 1 & !2] 0 {0 2}\n"
    + "[0 & !1 & !2] 0 {1 2}\n"
    + "[!0 & !1 & !2] 0 {0 1 2}\n"
    + "--END--";

  private static final String INPUT9 = "HOA: v1\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 2 \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 0 \n"
    + "[t] 0\n"
    + "[!0] 1 {0}"
    + "State: 1\n"
    + "[!1] 0 \n"
    + "[!0 & !1] 1 {0}"
    + "--END--";

  private static final String INPUT10 = "HOA: v1\n"
    + "States: 4\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 2 \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 0 \n"
    + "[t] 1 {0}\n"
    + "[0] 2 {0}\n"
    + "State: 2 \n"
    + "[0] 2 {0}\n"
    + "State: 3 \n"
    + "[t] 3 {0}\n"
    + "State: 1 \n"
    + "[1] 3 {0}\n"
    + "--END--";

  private static final String INPUT15 = "HOA: v1\n"
    + "tool: \"Owl\" \"* *\"\n"
    + "States: 4\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 3 \"a\" \"b\" \"c\"\n"
    + "--BODY--\n"
    + "State: 1\n"
    + "[0 & 2] 2 {0}\n"
    + "[!0 & 1 & 2] 1 {0}\n"
    + "[!0 & 2] 3\n"
    + "State: 0\n"
    + "[t] 0\n"
    + "[0] 2\n"
    + "[!0 & 1] 1\n"
    + "State: 3\n"
    + "[1] 1 {0}\n"
    + "[t] 3\n"
    + "State: 2\n"
    + "[0] 2 {0}\n"
    + "[!0 & 1] 1 {0}\n"
    + "[!0] 3\n"
    + "--END--";

  private static final String INPUT16 = "HOA: v1\n"
    + "tool: \"Owl\" \"* *\"\n"
    + "States: 5\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 3 \"c\" \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 1\n"
    + "[0 & 1 & !2] 2 {0}\n"
    + "[!0 & 1 & !2] 1 {0}\n"
    + "[!0 & 1 & !2] 3 {0}\n"
    + "[!0 & 1 & !2] 4\n"
    + "State: 0\n"
    + "[t] 0\n"
    + "[0] 2\n"
    + "[!0] 1\n"
    + "[!0] 3\n"
    + "State: 3\n"
    + "[0 & 2] 2 {0}\n"
    + "[!0 & 2] 1 {0}\n"
    + "[!0 & 2] 3 {0}\n"
    + "[!0 & 2] 4\n"
    + "State: 4\n"
    + "[0 & !2] 2 {0}\n"
    + "[!0 & !2] 3 {0}\n"
    + "[!0 & !2] 4\n"
    + "State: 2\n"
    + "[0] 2 {0}\n"
    + "[!0] 1 {0}\n"
    + "[!0] 3 {0}\n"
    + "[!0] 4\n"
    + "--END--";

  private static final String INPUT17 = "HOA: v1\n"
    + "tool: \"Owl\" \"* *\"\n"
    + "States: 2\n"
    + "Start: 0\n"
    + "acc-name: Buchi\n"
    + "Acceptance: 1 Inf(0)\n"
    + "properties: trans-acc trans-label \n"
    + "AP: 2 \"a\" \"b\"\n"
    + "--BODY--\n"
    + "State: 1\n"
    + "[0 & 1] 1 {0}\n"
    + "State: 0\n"
    + "[t] 0\n"
    + "[0 & 1] 1\n"
    + "--END--";

  @Test
  public void testApply() throws ParseException {
    runTest(INPUT, 6);
  }

  @Test
  public void testApply2() throws ParseException {
    runTest(INPUT2, 2);
  }

  @Test
  public void testApply3() throws ParseException {
    runTest(INPUT3, 12);
  }

  @Test
  public void testApply4() throws ParseException {
    runTest(INPUT4, 4);
  }

  @Test
  public void testApply5() throws ParseException {
    runTest(INPUT5, 4);
  }

  @Test
  public void testApply6() throws ParseException {
    runTest(INPUT6, 3);
  }

  @Test
  public void testApply7() throws ParseException {
    runTest(INPUT7, 6);
  }

  @Test
  public void testApply8() throws ParseException {
    runTest(INPUT8, 7);
  }

  @Test
  public void testApply9() throws ParseException {
    runTest(INPUT9, 3);
  }

  @Test
  public void testApply10() throws ParseException {
    runTest(INPUT10, 11);
  }

  @Test
  public void testApply15() throws ParseException {
    runTest(INPUT15, 23);
  }

  @Test
  public void testApply16() throws ParseException {
    runTest(INPUT16, 13);
  }

  @Test
  public void testApply17() throws ParseException {
    runTest(INPUT17, 3);
  }

  private void runTest(String input, int size) throws ParseException {
    Automaton<HoaState, GeneralizedBuchiAcceptance> nba = AutomatonReader.readHoa(input,
      DefaultEnvironment.annotated().factorySupplier(), GeneralizedBuchiAcceptance.class);
    nba.toHoa(new HOAIntermediateCheckValidity(new HOAConsumerNull()));

    NBA2DPAFunction<HoaState> translation = new NBA2DPAFunction<>();

    Automaton<?, ParityAcceptance> dpa = translation.apply(nba);
    dpa.toHoa(new HOAIntermediateCheckValidity(new HOAConsumerNull()));
    assertThat(dpa.size(), lessThanOrEqualTo(size));
  }
}
