package owl.translations.ltl2ldba.breakpointfree;

import owl.translations.ldba2dpa.Language;
import owl.translations.ldba2dpa.LanguageLattice;

public class BooleanLattice
  implements LanguageLattice<DegeneralizedBreakpointFreeState, FGObligations, Void> {

  private static final Language<Void> BOTTOM = new BottomLanguage();
  private static final Language<Void> TOP = new TopLanguage();

  @Override
  public Language<Void> getBottom() {
    return BOTTOM;
  }

  @Override
  public Language<Void> getLanguage(DegeneralizedBreakpointFreeState state) {
    return TOP;
  }

  @Override
  public Language<Void> getTop() {
    return TOP;
  }

  @Override
  public boolean isLivenessLanguage(FGObligations annotation) {
    return annotation.isPureLiveness();
  }

  @Override
  public boolean acceptsSafetyLanguage(DegeneralizedBreakpointFreeState state) {
    return isSafetyAnnotation(state.obligations) && state.liveness.isTrue();
  }

  @Override
  public boolean acceptsLivenessLanguage(DegeneralizedBreakpointFreeState state) {
    return isLivenessLanguage(state.obligations) && state.safety.isTrue();
  }

  @Override
  public boolean isSafetyAnnotation(FGObligations annotation) {
    return annotation.isPureSafety();
  }

  private static class BottomLanguage implements Language<Void> {
    @Override
    public Void getT() {
      return null;
    }

    @Override
    public boolean greaterOrEqual(Language<Void> language) {
      return language instanceof BottomLanguage;
    }

    @Override
    public boolean isBottom() {
      return true;
    }

    @Override
    public boolean isTop() {
      return false;
    }

    @Override
    public Language<Void> join(Language<Void> language) {
      return language;
    }
  }

  private static class TopLanguage implements Language<Void> {
    @Override
    public Void getT() {
      return null;
    }

    @Override
    public boolean greaterOrEqual(Language<Void> language) {
      return true;
    }

    @Override
    public boolean isBottom() {
      return false;
    }

    @Override
    public boolean isTop() {
      return true;
    }

    @Override
    public Language<Void> join(Language<Void> language) {
      return this;
    }
  }
}