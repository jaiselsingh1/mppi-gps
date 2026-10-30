"""The learner reserve cannot weaken or relabel actual ending success."""
import numpy as np
import pytest

from src.gps.bidirectional.credit import ending_training_metrics


def test_reserve_separates_physical_success_from_interior_preparation():
    threshold=-.1
    y=np.array([np.nextafter(threshold,-np.inf),np.nextafter(threshold,np.inf),-.101])
    physical=(y<=threshold).tolist()
    before=y.copy()
    zero=ending_training_metrics(y,threshold,0.)
    interior=ending_training_metrics(y,threshold,1e-4)
    assert physical==[True,False,True]
    assert zero['all_72_training_capture_flags']==physical
    assert interior['all_72_training_capture_flags']==[False,False,True]
    assert interior['worst_72_training_violation']>zero['worst_72_training_violation']
    assert np.array_equal(y,before) and (y<=threshold).tolist()==physical


@pytest.mark.parametrize('reserve',[-1.,np.nan,[1e-4],True])
def test_invalid_reserve_does_not_create_an_ending_target(reserve):
    with pytest.raises(ValueError):ending_training_metrics([-.2],-.1,reserve)
