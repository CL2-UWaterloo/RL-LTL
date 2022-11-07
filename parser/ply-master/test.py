from src.ply import lex
from src.ply import yacc

# -----------------------------------------------------------------------------
# using PLY To parse the following simple grammar.
#
#   expression : term DISJUNCTION term
#              | term CONJUNCTION term
#              | term
#
#   term       : term IMPLIES term
#              | term UNTIL term
#              | ALWAYS factor
#              | EVENTUALLY factor
#              | NEXT factor
#              | NEGATE factor
#              | factor
#
#   factor     : AP
#              | ALWAYS factor
#              | EVENTUALLY factor
#              | NEXT factor
#              | NEGATE factor
#              | LPAREN expression RPAREN
#
# -----------------------------------------------------------------------------

from src.ply.lex import lex
from src.ply.yacc import yacc

# --- Tokenizer

# All tokens must be named in advance.

tokens = ('ALWAYS', 'EVENTUALLY', 'UNTIL', 'IMPLIES', 'NEXT', 'NEGATE',
          'DISJUNCTION', 'CONJUNCTION', 'LPAREN', 'RPAREN','AP')

# Ignored characters
t_ignore = ' \t'

# Token matching rules are written as regexs
t_ALWAYS = r'\[\]'
t_EVENTUALLY = r'\<\>'
t_UNTIL = r'%'
t_IMPLIES = r'-\>'
t_NEXT = r'\>'
t_NEGATE = r'~'
t_DISJUNCTION = r'/\\'
t_CONJUNCTION = r'\\/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_AP = r'[a-zA-Z_][a-zA-Z0-9_]*'

# Ignored token with an action associated with it
def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')

# Error handler for illegal characters
def t_error(t):
    print(f'Illegal character {t.value[0]!r}')
    t.lexer.skip(1)

# Build the lexer object
lexer = lex()
    
# --- Parser

# Write functions for each grammar rule which is
# specified in the docstring.
def p_expression(p):
    '''
    expression :   term DISJUNCTION term
                 | term CONJUNCTION term
    '''
    # p is a sequence that represents rule contents.

    p[0] = (p[2], p[1], p[3])

def p_expression_term(p):
    '''
    expression : term
    '''
    p[0] = p[1]

def p_term(p):
    '''
    term : ALWAYS factor
        | EVENTUALLY factor
        | NEXT factor
        | NEGATE factor
    '''
    p[0] = (p[1], p[2])

def p_term_binary(p):
    '''
       term : term IMPLIES term
            | term UNTIL term
            | term DISJUNCTION term
            | term CONJUNCTION term
    '''
    p[0] = (p[2], p[1], p[3])

def p_term_factor(p):
    '''
    term : factor
    '''
    p[0] = p[1]

def p_factor_name(p):
    '''
    factor : AP
    '''
    p[0] = (None, p[1])

def p_factor_unary(p):
    '''
    factor : ALWAYS factor
           | EVENTUALLY factor
           | NEXT factor
           | NEGATE factor
    '''
    p[0] = (p[1], p[2])


def p_factor_grouped(p):
    '''
    factor : LPAREN expression RPAREN
    '''
    p[0] = p[2]

def p_error(p):
    print(f'Syntax error at {p.value!r}')

# Build the parser
parser = yacc()

# Parse an expression

t = "[] ( (~d) /\ ((b /\ ~ > b) -> >(~b % (a \/ c))) /\ (a -> >(~a % b)) /\ ((~b /\ >b /\ ~>>b)->(~a % c)) /\ (c->(~a % b)) /\ ((b /\>b)-><>a))"

ast = parser.parse(t)
print(ast)