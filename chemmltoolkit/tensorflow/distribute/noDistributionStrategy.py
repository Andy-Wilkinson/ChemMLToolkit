class NoDistributionStrategy:
    """ Distribution strategy the runs as if no strategy was applied.
    """
    def scope(self):
        return self

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return
