import keras.backend as K

from limelight.pkg_utils import reset_session


def test_session_reset():
    '''
    Test that resetted session meets the necessary requirements.
    '''
    reset_session()
    sess = K.get_session()

    assert hasattr(sess._config, 'allow_soft_placement')
    assert sess._config.allow_soft_placement

    assert hasattr(sess._config, 'gpu_options')
    assert hasattr(sess._config.gpu_options, 'allow_growth')
    assert sess._config.gpu_options.allow_growth
