from sympy import Eq, Symbol, solve

# Convert all numbers into floats if list contains a division
def chkfl(co_num_list):
    con_div = '/'
    final_con = []
    if con_div in co_num_list:
        for i in co_num_list:
            try:
                final_con.append(float(i))
            except (NameError, ValueError, SyntaxError, TypeError, ZeroDivisionError):
                final_con.append(i)
        final_con_str = [str(thing) for thing in final_con]
        exp_result = "".join(final_con_str)
    else:
        exp_result = "".join(co_num_list)
    return exp_result

# Convert labels into math operators
def convop(od_list_co):
    for n,i in enumerate(od_list_co):
        if i=='div':
            od_list_co[n]="/"
        elif i=='x':
            od_list_co[n]='*'
        elif i=='^':
            od_list_co[n]='**'
        elif i=='sqrt':
            od_list_co[n]='sqrt('
        elif i=='=':
            od_list_co[n]='=='
    return od_list_co

# Combine intergers between operators
def combint(od_list_co):
    co_num = ''
    co_num_list = []
    for n, i in enumerate(od_list_co):
        try:
            float(i)
            co_num += i
            if n == len(od_list_co)-1:
                co_num_list.append(co_num)
        except (NameError, ValueError, SyntaxError, TypeError, ZeroDivisionError):
            if co_num == '':
                co_num_list.append(i)
            else:
                co_num_list.append(co_num)
                co_num = ''
                co_num_list.append(i)
    return co_num_list

def getresult(co_num_list, exp_result):
    exp_split = exp_result.split("==")
    n = Symbol('n')
    if 'n' in co_num_list and '==' in co_num_list:
        try:
            eqn = Eq(eval(exp_split[0]), eval(exp_split[1]))
            result = solve(eqn)
        except (NameError, ValueError, SyntaxError, TypeError, ZeroDivisionError):
            result = '...'
    else:
        try:
            result = eval(exp_result)
        except (NameError, ValueError, SyntaxError, TypeError, ZeroDivisionError):
            result = '...'
    return result
