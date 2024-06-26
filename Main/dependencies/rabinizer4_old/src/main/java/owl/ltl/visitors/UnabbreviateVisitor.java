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

package owl.ltl.visitors;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.util.List;
import java.util.Set;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import owl.ltl.Conjunction;
import owl.ltl.Disjunction;
import owl.ltl.Formula;
import owl.ltl.GOperator;
import owl.ltl.MOperator;
import owl.ltl.ROperator;
import owl.ltl.UOperator;
import owl.ltl.WOperator;
import owl.run.modules.ImmutableTransformerParser;
import owl.run.modules.OwlModuleParser.TransformerParser;

public class UnabbreviateVisitor extends DefaultConverter {
  public static final TransformerParser CLI = ImmutableTransformerParser.builder()
    .key("unabbreviate")
    .optionsDirect(new Options()
      .addOption("w", "weak-until", false, "Remove W operator")
      .addOption("r", "release", false, "Remove R operator")
      .addOption("m", "strong-release", false, "Remove M operator")
    ).parser(settings -> {
      ImmutableList.Builder<Class<? extends Formula>> classes =
        ImmutableList.builder();
      if (settings.hasOption("weak-until")) {
        classes.add(WOperator.class);
      }

      if (settings.hasOption("release")) {
        classes.add(ROperator.class);
      }

      if (settings.hasOption("strong-release")) {
        classes.add(MOperator.class);
      }

      ImmutableList<Class<? extends Formula>> classList = classes.build();
      if (classList.isEmpty()) {
        throw new ParseException("No operation specified");
      }

      return DefaultConverter.asTransformer(new UnabbreviateVisitor(classList));
    })
    .build();

  // TODO Support for more operators
  private final Set<Class<? extends Formula>> classes;

  @SafeVarargs
  public UnabbreviateVisitor(Class<? extends Formula>... classes) {
    this.classes = ImmutableSet.copyOf(classes);
  }

  public UnabbreviateVisitor(List<Class<? extends Formula>> classes) {
    this.classes = ImmutableSet.copyOf(classes);
  }

  @Override
  public Formula visit(ROperator rOperator) {
    if (!classes.contains(ROperator.class)) {
      return super.visit(rOperator);
    }

    Formula left = rOperator.left.accept(this);
    Formula right = rOperator.right.accept(this);

    return Disjunction
      .of(GOperator.of(right), UOperator.of(right, Conjunction.of(left, right)));
  }

  @Override
  public Formula visit(WOperator wOperator) {
    if (!classes.contains(WOperator.class)) {
      return super.visit(wOperator);
    }

    Formula left = wOperator.left.accept(this);
    Formula right = wOperator.right.accept(this);

    return Disjunction.of(GOperator.of(left), UOperator.of(left, right));
  }

  @Override
  public Formula visit(MOperator mOperator) {
    if (!classes.contains(MOperator.class)) {
      return super.visit(mOperator);
    }

    Formula left = mOperator.left.accept(this);
    Formula right = mOperator.right.accept(this);

    return UOperator.of(right, Conjunction.of(left, right));
  }
}
