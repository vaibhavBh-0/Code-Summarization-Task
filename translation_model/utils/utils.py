from copy import deepcopy


def deepcopy_n(item, times=1):
    """
    :param item: item to be deep copied
    :param times: number of deep copies to be returned
    :return: A list of deep copies.
    """
    # May not work well on a TPU. So should be called outside of call
    # method of a Layer or Module.

    return [deepcopy(item) for _ in range(times)]


