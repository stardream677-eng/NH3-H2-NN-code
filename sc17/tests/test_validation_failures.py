import pytest
import validate_sc_v68 as val


def test_raw_feedback_failure_is_recorded_without_corrupting_other_modes():
    class InvalidEOS:
        def state_to_input_scaled(self,*args,**kwargs):raise ValueError('EOS cannot use negative mass fractions')
    failures=[]
    result=val.validation_feedback(InvalidEOS(),mode='pure_ar_raw',failures=failures,
                                   batch_index=0,endpoint=3,base_next=None)
    assert result is None
    assert failures[0]['mode']=='pure_ar_raw' and failures[0]['failure_step']==3
    assert 'negative' in failures[0]['message']
    with pytest.raises(ValueError):
        val.validation_feedback(InvalidEOS(),mode='pure_ar',failures=failures,
                                batch_index=0,endpoint=3,base_next=None)
