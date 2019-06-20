'''
Must return a string.
Must generate a random password.
The password must be exactly password_length characters long.
If has_symbols is True, the password must contain at least one symbol, such as #, !, or @.
If has_uppercase is True, the password must contain at least one uppercase letter.
The following criteria are optional, but will net you extra points:

The generator should support password_length < 1 000 000 characters.
The generator should not take more than 5 seconds to finish.
Update the function signature to take two more optional parameters, ignored_chars and allowed_chars.
The user can provide a list in either of these parameters to control which characters will be used to build the password.
Do not allow both lists to be passed at the same time. If this happens, raise a UserWarning explaining that only one may be passed.
Any characters in ignored_chars are guaranteed not to be used in the password.
Only characters in allowed_chars will be used in the password, if the list is present.
Update the docstring to explain how to use these new parameters.
'''



def generate_password(
    password_length: int = 8,
    has_symbols: bool = False,
    has_uppercase: bool = False,
    ignored_chars: list = None,
    allowed_chars: list = None
) -> str:
    """Generates a random password.

    The password will be exactly `password_length` characters.
    If `has_symbols` is True, the password will contain at least one symbol, such as #, !, or @.
    If `has_uppercase` is True, the password will contain at least one upper case letter.
    If 'ignored_chars' has entries, the password will ignore these characters within this list when creating the password.
    If 'allowed_chars' has entries, the password will only contain these characters within this list when creating the password.
    """
    #imports
    import string
    from random import random

    #create characters strings
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase
    #digits = string.digits
    symbols = string.punctuation
    
    #password variable
    password = ""

    #helper function to modify password if has_symbols or has_uppercase is used
    def passwordModifer(password, password_length, has_symbols, has_uppercase):
        #if has_symbols is true replace a random amount of characters with symbols
        if has_symbols:
            for i in range(1 + (int(random()* (password_length//1000)))):
                tmp = int(random()*password_length)
                password = password[:tmp] +  symbols[int(random()* len(symbols))] + password[tmp+1:]
        #if has_uppercase is true replace a random amount of characters with uppercase letters
        if has_uppercase:
            for i in range(1 + (int(random()* (password_length//1000)))):
                tmp = int(random()*password_length)
                password = password[:tmp] + upper[int(random()* len(upper))] + password[tmp+1:]

        return password
    
    #check if ignored_chars or allowed_char is used and raise exception if they both are used.
    if ignored_chars and allowed_chars:
        raise UserWarning('You may only use ignored_chars OR allowed_chars. You cannot use both options')
    #remove ignored_chars from string variables
    elif ignored_chars:
        for i in ignored_chars:
            if i.isupper():
                upper = upper.replace(i, '')
            elif i.islower():
                lower = lower.replace(i, '')
            elif i in symbols:
                symbols = symbols.replace(i, '')
            #elif i.isdigit():
                #digits = digits.replace(i, '')


    #create new string variables from allowed_chars      
    elif allowed_chars:
        lower = ''
        upper = ''
        #digits = ''
        symbols_a = ''
        allowed = ''
        for i in allowed_chars:
            if i.isupper():
                upper += i
            elif i.islower():
                lower+= i
            elif i in symbols:
                symbols_a += i
            #elif i.isdigit():
                #digits+= i
        if not lower:
            if len(upper) > 0:
                allowed = upper
            elif len(symbols_a) > 0:
                allowed = symbols_a
        else:
            allowed = lower

        password = ''.join([allowed[int(random()*len(allowed))] for _ in range(password_length)])
        password = passwordModifer(password, password_length, has_symbols, has_uppercase)
        return password
        
                
    #create password with all lowercase with len of password_length


    ''' More than double the time compared to random.random()
    for i in range(password_length):
        password += lower[randint(0,(len(lower) - 1))]
    '''

    '''
    #about the same as the listcomp 
    for i in range(password_length):
        password += lower[int(random()*len(lower))]
    '''

    '''
    #Fastest by far, but has numbers in it and not sure if digits are suppose to be included.
    import os
    tmp = os.urandom(password_length//2)
    password = tmp.hex()
    '''

    password = ''.join([lower[int(random()*len(lower))] for _ in range(password_length)])
    password = passwordModifer(password, password_length, has_symbols, has_uppercase)
    return password


   




