def named_module(name):
    # convenient function to unify all named imports
    from importlib import import_module
    return import_module(name)
