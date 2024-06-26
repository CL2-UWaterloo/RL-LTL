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

package owl.automaton.output;

import java.util.EnumSet;
import java.util.List;
import javax.annotation.Nullable;
import jhoafparser.consumer.HOAConsumer;

public interface HoaPrintable {
  @Nullable
  default String getName() {
    return null;
  }

  List<String> getVariables();

  // rename feedTo...
  default void toHoa(HOAConsumer consumer) {
    toHoa(consumer, EnumSet.noneOf(HoaOption.class));
  }

  void toHoa(HOAConsumer consumer, EnumSet<HoaOption> options);

  enum HoaOption {
    /**
     * Print annotations, e.g. state labels, if available
     */
    ANNOTATIONS,
    /**
     * Create one transition for each element of the AP-power-set instead of complex expressions
     * (which are not supported by all parsers).
     */
    SIMPLE_TRANSITION_LABELS
  }
}
