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

import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import org.junit.Before;
import org.junit.Test;
import owl.factories.EquivalenceClassFactory;
import owl.ltl.parser.LtlParser;
import owl.ltl.rewriter.RewriterFactory;
import owl.ltl.rewriter.RewriterFactory.RewriterEnum;

public abstract class EquivalenceClassTest {
  private static final List<String> formulaStrings = ImmutableList
    .of("G a", "F G a", "G a | G b", "(G a) U (G b)", "X G b", "F F ((G a) & b)", "a & G b");
  private static final List<LabelledFormula> formulas = formulaStrings.stream()
    .map(LtlParser::parse)
    .collect(ImmutableList.toImmutableList());
  private Formula contradiction;
  private EquivalenceClassFactory factory;
  private Formula literal;
  private Formula tautology;

  @Before
  public void setUp() {
    contradiction = BooleanConstant.FALSE;
    tautology = BooleanConstant.TRUE;
    literal = new Literal(0);

    factory = setUpFactory(LtlParser.parse("a & b & c & d"));
  }

  public abstract EquivalenceClassFactory setUpFactory(LabelledFormula domain);

  @Test
  public void testEmptyDomain() {
    EquivalenceClassFactory factory = setUpFactory(LabelledFormula.of(BooleanConstant.TRUE,
      List.of()));
    assertNotEquals(factory, null);
  }

  @Test
  public void testEqualsAndHashCode() {
    Collection<EquivalenceClass> classes = new ArrayList<>();

    classes.add(factory.of(contradiction));
    classes.add(factory.of(tautology));
    classes.add(factory.of(literal));
    classes.add(factory.of(new Disjunction(tautology, contradiction, literal)));
    classes.add(factory.of(new Conjunction(tautology, contradiction, literal)));

    for (EquivalenceClass lhs : classes) {
      for (EquivalenceClass rhs : classes) {
        assertEquals(lhs.equals(rhs), lhs.equals(rhs));

        if (lhs.equals(rhs)) {
          assertEquals(lhs.hashCode(), rhs.hashCode());
        }
      }
    }
  }

  @Test
  public void testEquivalent() {
    EquivalenceClass equivalenceClass = factory.of(contradiction);

    assertEquals(equivalenceClass, equivalenceClass);
    assertEquals(equivalenceClass, factory.of(RewriterFactory
      .apply(RewriterEnum.MODAL_ITERATIVE, new Conjunction(literal, new Literal(0, true)))));
  }

  @Test
  public void testExistsAndSat() {
    Predicate<Formula> predicate = ((Predicate<Formula>) GOperator.class::isInstance).negate();

    Formula[] formulas = {
      LtlParser.syntax("a"),
      LtlParser.syntax("G a"),
      LtlParser.syntax("G a | G b | a"),
      LtlParser.syntax("G a & G b & a"),
    };

    EquivalenceClass classA = factory.of(formulas[0]);
    EquivalenceClass classExistsA = classA.exists(predicate);
    assertEquals(factory.getTrue(), classExistsA);

    EquivalenceClass classB = factory.of(formulas[1]);
    EquivalenceClass classExistsB = classB.exists(predicate);
    assertEquals(classB, classExistsB);

    EquivalenceClass classC = factory.of(formulas[2]);
    EquivalenceClass classExistsC = classC.exists(predicate);
    assertEquals(factory.getTrue(), classExistsC);

    EquivalenceClass classD = factory.of(formulas[3]);
    EquivalenceClass classExistsD = classD.exists(predicate);
    assertEquals(factory.of(LtlParser.syntax("G a & G b")), classExistsD);
  }

  @Test
  public void testFrequencyGNotFalse() {
    LabelledFormula formula = LtlParser.parse("G { >= 0.4} a");
    EquivalenceClassFactory factory = setUpFactory(formula);
    EquivalenceClass clazz = factory.of(formula.formula);
    assertNotEquals(factory.getFalse(), clazz.unfold().temporalStep(new BitSet(0)));
  }

  @Test
  public void testGetAtoms() {
    LabelledFormula formula = LtlParser.parse("a & (a | b) & (F c)");
    EquivalenceClassFactory factory = setUpFactory(formula);
    EquivalenceClass clazz = factory.of(formula.formula);
    BitSet atoms = new BitSet();
    atoms.set(0);
    assertThat(clazz.getAtoms(), is(atoms));
    atoms.set(2);
    assertThat(clazz.unfold().getAtoms(), is(atoms));
  }

  @Test
  public void testGetAtoms2() {
    LabelledFormula formula = LtlParser.parse("(a | (b & X a) | (F a)) & (c | (b & X a) | (F a))");
    EquivalenceClassFactory factory = setUpFactory(formula);
    EquivalenceClass clazz = factory.of(formula.formula);
    BitSet atoms = new BitSet();
    atoms.set(0, 3);
    assertEquals(atoms, clazz.getAtoms());
  }

  @Test
  public void testGetAtomsEmpty() {
    LabelledFormula formula = LtlParser.parse("G a");
    EquivalenceClassFactory factory = setUpFactory(formula);
    EquivalenceClass clazz = factory.of(formula.formula);
    BitSet atoms = new BitSet();
    assertEquals(atoms, clazz.getAtoms());
    atoms.set(0);
    assertEquals(atoms, clazz.unfold().getAtoms());
  }

  @Test
  public void testGetRepresentative() {
    assertEquals(contradiction, factory.of(contradiction).getRepresentative());
  }

  @Test
  public void testGetSupport() {
    Formula[] formulas = {
      LtlParser.syntax("a"),
      LtlParser.syntax("F a"),
      LtlParser.syntax("G a")
    };

    EquivalenceClass clazz = factory.of(Conjunction.of(formulas));
    assertEquals(Set.of(formulas), clazz.getSupport());
  }

  @Test
  public void testImplies() {
    EquivalenceClass contradictionClass = factory.of(contradiction);
    EquivalenceClass tautologyClass = factory.of(tautology);
    EquivalenceClass literalClass = factory.of(literal);

    assertTrue(contradictionClass.implies(contradictionClass));

    assertTrue(contradictionClass.implies(tautologyClass));
    assertTrue(contradictionClass.implies(literalClass));

    assertTrue(literalClass.implies(tautologyClass));
    assertFalse(literalClass.implies(contradictionClass));

    assertFalse(tautologyClass.implies(contradictionClass));
    assertFalse(tautologyClass.implies(literalClass));
  }

  // @Test
  public void testLtlBackgroundTheory1() {
    Formula f1 = LtlParser.syntax("G p0 & p0");
    Formula f2 = LtlParser.syntax("G p0");
    assertEquals(f2, RewriterFactory.apply(RewriterEnum.MODAL_ITERATIVE, f1));
  }

  // @Test
  public void testLtlBackgroundTheory2() {
    Formula f1 = LtlParser.syntax("G p0 | p0");
    Formula f2 = LtlParser.syntax("p0");
    assertEquals(f2, RewriterFactory.apply(RewriterEnum.MODAL_ITERATIVE, f1));
  }

  // @Test
  public void testLtlBackgroundTheory3() {
    Formula f1 = new Literal(1, false);
    Formula f2 = new GOperator(f1);
    Formula f5 = RewriterFactory.apply(RewriterEnum.MODAL, new Conjunction(
      new GOperator(new FOperator(new XOperator(f1))), f2));
    assertEquals(RewriterFactory.apply(RewriterEnum.MODAL_ITERATIVE, f5), f2);
  }

  @Test
  public void testSubstitute() {
    EquivalenceClass[] formulas = {
      factory.of(LtlParser.syntax("a")),
      factory.of(LtlParser.syntax("G a")),
      factory.of(LtlParser.syntax("G a & a"))
    };

    assertEquals(formulas[1].substitute(Formula::unfold), formulas[2]);
    assertEquals(formulas[2].substitute(x -> x instanceof GOperator ? BooleanConstant.TRUE : x),
      formulas[0]);
  }

  @SuppressWarnings("ReuseOfLocalVariable")
  @Test
  public void testTemporalStep() {
    BitSet stepSet = new BitSet();
    LabelledFormula formula = LtlParser.parse("a & X (! a)");
    EquivalenceClassFactory factory = setUpFactory(formula);
    assertEquals(factory.of(LtlParser.syntax("! a")),
      factory.of(LtlParser.syntax("X ! a")).temporalStep(stepSet));
    assertEquals(factory.of(LtlParser.syntax("a")),
      factory.of(LtlParser.syntax("X a")).temporalStep(stepSet));

    formula = LtlParser.parse("(! a) & X (a)");
    factory = setUpFactory(formula);
    assertEquals(factory.of(LtlParser.syntax("! a")),
      factory.of(LtlParser.syntax("X ! a")).temporalStep(stepSet));
    assertEquals(factory.of(LtlParser.syntax("a")),
      factory.of(LtlParser.syntax("X a")).temporalStep(stepSet));

    formula = LtlParser.parse("(a) & X (a)");
    factory = setUpFactory(formula);
    assertEquals(factory.of(LtlParser.syntax("! a")),
      factory.of(LtlParser.syntax("X ! a")).temporalStep(stepSet));
    assertEquals(factory.of(LtlParser.syntax("a")),
      factory.of(LtlParser.syntax("X a")).temporalStep(stepSet));

    formula = LtlParser.parse("(! a) & X (! a)");
    factory = setUpFactory(formula);
    assertEquals(factory.of(LtlParser.syntax("! a")),
      factory.of(LtlParser.syntax("X ! a")).temporalStep(stepSet));
    assertEquals(factory.of(LtlParser.syntax("a")),
      factory.of(LtlParser.syntax("X a")).temporalStep(stepSet));
  }

  @Test
  public void testUnfoldUnfold() {
    for (LabelledFormula formula : formulas) {
      EquivalenceClassFactory factory = setUpFactory(formula);
      EquivalenceClass ref = factory.of(formula.formula.unfold());
      EquivalenceClass clazz = factory.of(formula.formula).unfold();
      assertEquals(ref, clazz);
      assertEquals(clazz, clazz.unfold());
    }
  }

  @Test
  public void testRewrite() {
    Formula formula = LtlParser.syntax("G (a | X!b) | F c");
    EquivalenceClass clazz = factory.of(formula).unfold();

    Function<Formula, Formula> substitution = x -> {
      if (x instanceof FOperator) {
        return BooleanConstant.FALSE;
      }

      if (x instanceof Literal) {
        return ((Literal) x).getAtom() == 2 ? x : BooleanConstant.TRUE;
      }

      return BooleanConstant.TRUE;
    };

    EquivalenceClass core = clazz.substitute(substitution);
    assertThat(core, is(factory.getTrue()));

    clazz = clazz.temporalStepUnfold(new BitSet());

    core = clazz.substitute(substitution);
    assertThat(core, is(factory.getTrue()));
  }
}