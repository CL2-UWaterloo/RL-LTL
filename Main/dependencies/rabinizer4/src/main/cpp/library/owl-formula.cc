#include "owl-formula.h"
#include "owl-private.h"

namespace owl {
    Formula::Formula(JNIEnv *env, jobject formula) : env(env), formula(formula) {}

    Formula::Formula(const Formula &formula) : env(formula.env), formula(ref(formula.env, formula.formula)) {}

    Formula::Formula(const Formula &&formula) noexcept : env(formula.env), formula(ref(formula.env, formula.formula)) {}

    Formula::~Formula() {
       deref(env, formula);
    }

    void Formula::print() const {
        jclass clazz = get_class(env, formula);
        jmethodID toString = get_methodID(env, clazz, "toString", "()Ljava/lang/String;");
        auto string = call_method<jstring>(env, formula, toString);

        // Get a C-style string
        const char* str = env->GetStringUTFChars(string, JNI_FALSE);
        std::cout << str << std::endl;
        env->ReleaseStringUTFChars(string, str);

        deref(env, clazz, string);
    }

    void FormulaFactory::bind_static(int index, const char *className, const char *methodName, const char *signature) {
        bind_static_method(env, className, methodName, signature, classes[index], methodIDs[index]);
    }

    FormulaFactory::FormulaFactory(JNIEnv* env) : env(env) {
        const char* of = "of";
        const char* unarySignature = "(Lowl/ltl/Formula;)Lowl/ltl/Formula;";
        const char* binarySignature = "(Lowl/ltl/Formula;Lowl/ltl/Formula;)Lowl/ltl/Formula;";

        bind_static(BooleanConstant, "owl/ltl/BooleanConstant", of, "(Z)Lowl/ltl/BooleanConstant;");
        bind_static(Literal, "owl/ltl/Literal", of, "(IZ)Lowl/ltl/Literal;");

        bind_static(Conjunction, "owl/ltl/Conjunction", of, binarySignature);
        bind_static(Disjunction, "owl/ltl/Disjunction", of, binarySignature);

        bind_static(FOperator, "owl/ltl/FOperator", of, unarySignature);
        bind_static(GOperator, "owl/ltl/GOperator", of, unarySignature);
        bind_static(XOperator, "owl/ltl/XOperator", of, unarySignature);

        bind_static(UOperator, "owl/ltl/UOperator", of, binarySignature);
        bind_static(ROperator, "owl/ltl/ROperator", of, binarySignature);
        bind_static(WOperator, "owl/ltl/WOperator", of, binarySignature);
        bind_static(MOperator, "owl/ltl/MOperator", of, binarySignature);

        bind_static_method(env, "owl/ltl/parser/LtlParser", "syntax", "(Ljava/lang/String;Ljava/util/List;)Lowl/ltl/Formula;", parser, parseID);
        bind_static_method(env, "owl/ltl/parser/TlsfParser", "parse", "(Ljava/lang/String;)Lowl/ltl/tlsf/Tlsf;", tlsfParser, tlsfParseID);
    }

    FormulaFactory::~FormulaFactory() {
        for (auto& clazz : classes) {
            deref(env, clazz);
        }

        deref(env, parser);
        deref(env, tlsfParser);
    }

    template<typename... Args>
    Formula FormulaFactory::create(int index, Args... args) {
        return Formula(env, call_static_method<jobject>(env, classes[index], methodIDs[index], args...));
    }

    Formula FormulaFactory::createFOperator(const Formula& formula) {
        return create<jobject>(FOperator, formula.formula);
    }

    Formula FormulaFactory::createGOperator(const Formula& formula) {
        return create<jobject>(GOperator, formula.formula);
    }

    Formula FormulaFactory::createXOperator(const Formula& formula) {
        return create<jobject>(XOperator, formula.formula);
    }

    Formula FormulaFactory::createUOperator(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(UOperator, left.formula, right.formula);
    }

    Formula FormulaFactory::createROperator(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(ROperator, left.formula, right.formula);
    }

    Formula FormulaFactory::createMOperator(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(MOperator, left.formula, right.formula);
    }

    Formula FormulaFactory::createWOperator(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(WOperator, left.formula, right.formula);
    }

    Formula FormulaFactory::createConjunction(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(Conjunction, left.formula, right.formula);
    }

    Formula FormulaFactory::createDisjunction(const Formula& left, const Formula& right) {
        return create<jobject, jobject>(Disjunction, left.formula, right.formula);
    }

    Formula FormulaFactory::createConstant(const bool value) {
        return create<bool>(BooleanConstant, value);
    }

    Formula FormulaFactory::createLiteral(const int atom) {
        return create<int, bool>(Literal, atom, false);
    }

    Formula FormulaFactory::createNegatedLiteral(const int atom) {
        return create<int, bool>(Literal, atom, true);
    }

    Formula FormulaFactory::createImplication(const Formula &left, const Formula &right) {
        Formula notLeft = createNegation(left);
        return createDisjunction(notLeft, right);
    }

    Formula FormulaFactory::createBiimplication(const Formula &left, const Formula &right) {
        return createConjunction(createImplication(left, right), createImplication(right, left));
    }

    Formula FormulaFactory::createNegation(const Formula &formula) {
        jclass clazz;
        jmethodID notID = get_methodID(env, formula.formula, clazz, "not", "()Lowl/ltl/Formula;");
        Formula negation = Formula(env, call_method<jobject>(env, formula.formula, notID));
        deref(env, clazz);
        return negation;
    }

    Formula FormulaFactory::parse(const std::string &formula_string, const std::vector<std::string>& apMapping) {
        jstring string = copy_to_java(env, formula_string);
        jobject mapping = copy_to_java(env, apMapping);
        auto formula = call_static_method<jobject, jstring, jobject>(env, parser, parseID, string, mapping);
        deref(env, string, mapping);
        return Formula(env, formula);
    }

    Formula FormulaFactory::parseTlsf(const std::string &tlsf_string, std::vector<std::string> &apMapping,
                                      int &numberOfInputVariables) {
        jstring string = copy_to_java(env, tlsf_string);
        auto tlsf = call_static_method<jobject, jstring>(env, tlsfParser, tlsfParseID, string);
        auto labelled_formula = call_method<jobject>(env, tlsf, "toFormula", "()Lowl/ltl/LabelledFormula;");
        auto formula = get_object_field<jobject>(env, labelled_formula, "formula", "Lowl/ltl/Formula;");
        auto variables = get_object_field<jobject>(env, labelled_formula, "variables", "Lcom/google/common/collect/ImmutableList;");

        apMapping = copy_from_java(env, variables);
        numberOfInputVariables = call_int_method<>(env, tlsf, "numberOfInputs", "()I");
        deref(env, tlsf, labelled_formula);

        return Formula(env, formula);
    }

    FormulaRewriter::FormulaRewriter(JNIEnv *env) : env(env) {
        bind_static_method(env, "owl/ltl/rewriter/ShiftRewriter", "shiftLiterals",
                           "(Lowl/ltl/Formula;)Lowl/ltl/rewriter/ShiftRewriter$ShiftedFormula;", shift_rewriter,
                           shift_literalsID);
        bind_static_method(env, "owl/ltl/rewriter/RealizabilityRewriter", "split",
                           "(Lowl/ltl/Formula;ILjava/util/Map;)[Lowl/ltl/Formula;", realizability_rewriter, splitID);
        bind_static_method(env, "owl/ltl/rewriter/RewriterFactory", "apply", "(Lowl/ltl/Formula;)Lowl/ltl/Formula;", simplifier, simplifyID);
    }

    std::vector<Formula> FormulaRewriter::split(const Formula &input, int numberOfInputVariables, std::map<int, bool>& map) {
        auto java_map = new_object<jobject>(env, "java/util/HashMap", "()V");
        auto array = call_static_method<jobjectArray>(env, realizability_rewriter, splitID, input.formula, numberOfInputVariables, java_map);
        map = copy_from_java(env, java_map);

        jsize length = env->GetArrayLength(array);
        std::vector<Formula> formulas = std::vector<Formula>();

        for (int i = 0; i < length; ++i) {
            jobject output = make_global(env, env->GetObjectArrayElement(array, i));
            formulas.emplace_back(Formula(env, output));
        }

        deref(env, array);
        return formulas;
    }

    Formula FormulaRewriter::shift_literals(const Formula &formula, std::map<int, int> &map) {
        map.clear();

        auto shifted_formula = call_static_method<jobject>(env, shift_rewriter, shift_literalsID, formula.formula);
        auto result = get_object_field<jobject>(env, shifted_formula, "formula", "Lowl/ltl/Formula;");
        auto mapping = get_object_field<jintArray>(env, shifted_formula, "mapping", "[I");

        jsize length = env->GetArrayLength(mapping);

        for (int i = 0; i < length; ++i) {
            int j;
            env->GetIntArrayRegion(mapping, i, 1, &j);

            if (j != -1) {
                map[i] = j;
            }
        }

        deref(env, shifted_formula, mapping);
        return Formula(env, result);
    }

    Formula FormulaRewriter::simplify(const Formula &formula) {
        return Formula(env, call_static_method<jobject, jobject>(env, simplifier, simplifyID, formula.formula));
    }

    FormulaRewriter::~FormulaRewriter() {
        deref(env, shift_rewriter, realizability_rewriter, simplifier);
    }
}