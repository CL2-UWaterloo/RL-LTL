package owl.run.parser;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import owl.run.Pipeline;
import owl.run.modules.OwlModuleRegistry;
import owl.run.modules.OwlModuleRegistry.OwlModuleNotFoundException;
import owl.run.modules.OwlModuleRegistry.Type;

public final class OwlParser {
  private static final Logger logger = Logger.getLogger(PipelineParser.class.getName());
  public final Pipeline pipeline;
  public final CommandLine globalSettings;

  public OwlParser(Pipeline pipeline, CommandLine globalSettings) {
    this.pipeline = pipeline;
    this.globalSettings = globalSettings;
  }

  @SuppressWarnings("NestedTryStatement")
  @Nullable
  public static OwlParser parse(String[] arguments, CommandLineParser parser,
    Options globalOptions, OwlModuleRegistry registry) {
    logger.log(Level.FINE, "Parsing arguments list {0}", Arrays.toString(arguments));
    if (arguments.length == 0 || ParseUtil.isHelp(arguments)) {
      ParseUtil.println("This is owl. Owl is a flexible "
        + "tool for various translations involving automata. To allow for great flexibility and "
        + "rapid prototyping, it was equipped with a very flexible module-based command line "
        + "interface. You can specify a specific translation in the following way:\n"
        + '\n'
        + "  owl <settings> <input parser> --- <multiple modules> --- <output>\n"
        + '\n'
        + "Available settings for registered modules are printed below");

      ParseUtil.printHelp("Global settings:", globalOptions);
      ParseUtil.println();
      ParseUtil.printList(Type.READER,
        ParseUtil.getSortedSettings(registry, Type.READER), null);
      ParseUtil.println();
      ParseUtil.printList(Type.TRANSFORMER,
        ParseUtil.getSortedSettings(registry, Type.TRANSFORMER), null);
      ParseUtil.println();
      ParseUtil.printList(Type.WRITER,
        ParseUtil.getSortedSettings(registry, Type.WRITER), null);
      return null;
    }

    CommandLine globalSettings;
    try {
      globalSettings = parser.parse(globalOptions, arguments, true);
    } catch (ParseException e) {
      ParseUtil.printHelp("global", globalOptions, e.getMessage());
      return null;
    }

    List<PipelineParser.ModuleDescription> split =
      PipelineParser.split(globalSettings.getArgList(), "---"::equals);

    Pipeline pipeline;
    try {
      pipeline = PipelineParser.parse(split, parser, registry);
    } catch (OwlModuleNotFoundException e) {
      ParseUtil.printList(e.type, registry.getSettings(e.type), e.name);
      return null;
    } catch (PipelineParser.ModuleParseException e) {
      ParseUtil.printModuleHelp(e.settings, e.getMessage());
      return null;
    } catch (IllegalArgumentException e) {
      System.err.println(e.getMessage()); // NOPMD
      return null;
    }
    globalSettings.getArgList().clear();

    return new OwlParser(pipeline, globalSettings);
  }
}
