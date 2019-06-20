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
    symbols = string.punctuation
    
    #password variable
    password = ""
    
    #helper function
    def passwordModifer(password, password_length, allowed, has_symbols, has_uppercase):
        """Helper function to modify a password if the has_symbol or has_uppercase variables are set to True.
        The allowed variable is the string of characters that were used to make the generated password.
        Either lower if no allowed_chars were used or allowed if allowed_chars were used.
        This is used to make sure no added symbols are removed when adding uppercase as well.
        """
        
        #modifier based on the password_length to loop through to add new symbols or uppercase characters
        if password_length > 100000:
            modifier = password_length // 5000
        elif password_length > 10000:
            modifier = password_length // 1000
        elif password_length > 1000:
             modifier = password_length // 10
        elif password_length > 100:
            modifier = password_length // 5
        else:
            modifier = password_length // 2

        #if has_symbols is true replace a random amount of characters with symbols
        if has_symbols:
            try:
                for i in range(1 + (int(random()* (modifier)))):
                    tmp = int(random()*password_length)
                    password = password[:tmp] +  symbols[int(random()* len(symbols))] + password[tmp+1:]
            #In case allowed_chars were used along with has_symbols, but no symbols characters were included in allowed_chars
            except:
                raise UserWarning('You have used allowed characters and has_uppercase as True, but did not include any uppercase characters!')
            

            
        #if has_uppercase is true replace a random amount of characters with uppercase letters
        if has_uppercase:
            try:
                for i in range(1 + (int(random()* (modifier)))):
                    tmp = int(random()*password_length)
                    #Don't accidently remove the new symbol characters to replace them with an uppercase letter
                    while password[tmp] not in allowed:
                        tmp = int(random()*password_length)
                    password = password[:tmp] + upper[int(random()* len(upper))] + password[tmp+1:]
            #In case allowed_chars were used along with has_uppercase, but no uppercase characters were included in allowed_chars
            except:
                raise UserWarning('You have used allowed characters and has_uppercase as True, but did not include any uppercase characters!')

        

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

    #create new string variables from allowed_chars      
    elif allowed_chars:
        lower = ''
        upper = ''
        symbols_a = ''
        allowed = ''
        for i in allowed_chars:
            if i.isupper():
                upper += i
            elif i.islower():
                lower+= i
            elif i in symbols:
                symbols_a += i
        symbols = symbols_a
        if not lower:
            if len(upper) > 0:
                allowed = upper
            elif len(symbols_a) > 0:
                allowed = symbols_a
        else:
            allowed = lower

        #create password from allowed characters with len of password_length
        password = ''.join([allowed[int(random()*len(allowed))] for _ in range(password_length)])
        #if has_symbols or has_uppercase modify the password with the helper function
        password = passwordModifer(password, password_length, allowed, has_symbols, has_uppercase)
        return password
        

    #create password with all lowercase characters with len of password_length
    password = ''.join([lower[int(random()*len(lower))] for _ in range(password_length)])
    #if has_symbols or has_uppercase modify the password with the helper function
    password = passwordModifer(password, password_length, lower, has_symbols, has_uppercase)
    return password







