import pytest

from vid2script.agent import batch_end


@pytest.mark.parametrize("start,stop,batch,expected", [
    (0, 30, 30, 30),
    (0, 31, 30, 29),
    (0, 31.5, 30, 29.5),
    (0, 11, 10, 11),
    (0, 12, 10, 10),
    (0, 32, 30, 30),
    (0, 3, 2, 3),
    (0, 4, 3, 4),
])
def test_batch_end_never_leaves_a_too_short_tail(start, stop, batch, expected):
    end = batch_end(start, stop, batch, 2, 30)
    assert end == expected
    assert stop == end or stop - end >= 2
