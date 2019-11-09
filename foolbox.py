from __future__ import absolute_import
from __future__ import division

import numpy as np
import sys
import abc
abstractmethod = abc.abstractmethod
import collections
import logging
import numpy as np
from collections import Iterable
import warnings
import functools
import numbers
from numbers import Number


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})
    
def _create_preprocessing_fn(params):
    if isinstance(params, collections.Mapping):
        mean = params.get("mean", 0)
        std = params.get("std", 1)
        axis = params.get("axis", None)
    else:
        mean, std = params
        axis = None

    mean = np.asarray(mean)
    std = np.asarray(std)

    mean = np.atleast_1d(mean)
    std = np.atleast_1d(std)

    if axis is not None:
        assert mean.ndim == 1, "If axis is specified, mean should be 1-dimensional"
        assert std.ndim == 1, "If axis is specified, std should be 1-dimensional"
        assert (
            axis < 0
        ), "axis must be negative integer, with -1 representing the last axis"
        s = (1,) * (abs(axis) - 1)
        mean = mean.reshape(mean.shape + s)
        std = std.reshape(std.shape + s)

    def identity(x):
        return x

    if np.all(mean == 0) and np.all(std == 1):

        def preprocessing(x):
            return x, identity

    elif np.all(std == 1):

        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            return x - _mean, identity

    elif np.all(mean == 0):

        def preprocessing(x):
            _std = std.astype(x.dtype)

            def grad(dmdp):
                return dmdp / _std

            return x / _std, grad

    else:

        def preprocessing(x):
            _mean = mean.astype(x.dtype)
            _std = std.astype(x.dtype)
            result = x - _mean
            result /= _std

            def grad(dmdp):
                return dmdp / _std

            return result, grad

    return preprocessing

    
class Model(ABC):
    """Base class to provide attacks with a unified interface to models.
    The :class:`Model` class represents a model and provides a
    unified interface to its predictions. Subclasses must implement
    batch_predictions and num_classes.
    :class:`Model` instances can be used as context managers and subclasses
    can require this to allocate and release resources.
    Parameters
    ----------
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    """

    def __init__(self, bounds, channel_axis, preprocessing=(0, 1)):
        assert len(bounds) == 2
        self._bounds = bounds
        self._channel_axis = channel_axis

        if not callable(preprocessing):
            preprocessing = _create_preprocessing_fn(preprocessing)
        assert callable(preprocessing)
        self._preprocessing = preprocessing

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def bounds(self):
        return self._bounds

    def channel_axis(self):
        return self._channel_axis

    def _process_input(self, x):
        p, grad = self._preprocessing(x)
        if hasattr(p, 'dtype'):
            assert p.dtype == x.dtype
        p = np.asarray(p, dtype=x.dtype)
        assert callable(grad)
        return p, grad

    def _process_gradient(self, backward, dmdp):
        """
        backward: `callable`
            callable that backpropagates the gradient of the model w.r.t to
            preprocessed input through the preprocessing to get the gradient
            of the model's output w.r.t. the input before preprocessing
        dmdp: gradient of model w.r.t. preprocessed input
        """
        if backward is None:  # pragma: no cover
            raise ValueError('Your preprocessing function does not provide'
                             ' an (approximate) gradient')
        print('going through process gradient')
        print('calling backward with array of shape: {}'.format(dmdp.shape))
        dmdx = backward(dmdp)
        dmdx = dmdx.astype('float32')
        assert dmdx.dtype == dmdp.dtype
        return dmdx

    @abstractmethod
    def batch_predictions(self, images):
        """Calculates predictions for a batch of images.
        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of inputs with shape as expected by the model.
        Returns
        -------
        `numpy.ndarray`
            Predictions (logits, i.e. before the softmax) with shape
            (batch size, number of classes).
        See Also
        --------
        :meth:`predictions`
        """
        raise NotImplementedError

    def predictions(self, image):
        """Convenience method that calculates predictions for a single image.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        Returns
        -------
        `numpy.ndarray`
            Vector of predictions (logits, i.e. before the softmax) with
            shape (number of classes,).
        See Also
        --------
        :meth:`batch_predictions`
        """
        return np.squeeze(self.batch_predictions(image[np.newaxis]), axis=0)

    @abstractmethod
    def num_classes(self):
        """Determines the number of classes.
        Returns
        -------
        int
            The number of classes for which the model creates predictions.
        """
        raise NotImplementedError



class DifferentiableModel(Model):
    """Base class for differentiable models that provide gradients.
    The :class:`DifferentiableModel` class can be used as a base
    class for models that provide gradients. Subclasses must implement
    predictions_and_gradient.
    A model should be considered differentiable based on whether it
    provides a :meth:`predictions_and_gradient` method and a
    :meth:`gradient` method, not based on whether it subclasses
    :class:`DifferentiableModel`.
    A differentiable model does not necessarily provide reasonable
    values for the gradients, the gradient can be wrong. It only
    guarantees that the relevant methods can be called.
    """

    @abstractmethod
    def predictions_and_gradient(self, image, label):
        """Calculates predictions for an image and the gradient of
        the cross-entropy loss w.r.t. the image.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        label : int
            Reference label used to calculate the gradient.
        Returns
        -------
        predictions : `numpy.ndarray`
            Vector of predictions (logits, i.e. before the softmax) with
            shape (number of classes,).
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.
        See Also
        --------
        :meth:`gradient`
        """
        raise NotImplementedError

    def gradient(self, image, label):
        """Calculates the gradient of the cross-entropy loss w.r.t. the image.
        The default implementation calls predictions_and_gradient.
        Subclasses can provide more efficient implementations that
        only calculate the gradient.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        label : int
            Reference label used to calculate the gradient.
        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient of the cross-entropy loss w.r.t. the image. Will
            have the same shape as the image.
        See Also
        --------
        :meth:`gradient`
        """
        _, gradient = self.predictions_and_gradient(image, label)
        return gradient

    @abstractmethod
    def backward(self, gradient, image):
        """Backpropagates the gradient of some loss w.r.t. the logits
        through the network and returns the gradient of that loss w.r.t
        to the input image.
        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the image.
        See Also
        --------
        :meth:`gradient`
        """
        raise NotImplementedError

class KerasModel(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Keras` model.
    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        from keras import backend as K
        import keras
        from pkg_resources import parse_version

        assert parse_version(keras.__version__) >= parse_version('2.0.7'), 'Keras version needs to be 2.0.7 or newer'  # noqa: E501

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        if predicts == 'probabilities':
            if K.backend() == 'tensorflow':
                predictions, = predictions.op.inputs
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=True)
            else:
                logging.warning('relying on numerically unstable conversion'
                                ' from probabilities to softmax')
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=False)

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)
        grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            grad_loss_input = K.reshape(grad_loss_input, (1,) + grad_loss_input.shape)  # noqa: E501
        else:
            assert not isinstance(grads, list)
            grad = grads

            grad_loss_input = grads_loss_input

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
            [images_input, label_input],
            [predictions, grad])
        self._bw_grad_fn = K.function(
            [grad_loss_output, images_input],
            [grad_loss_input])

    def _to_logits(self, predictions):
        from keras import backend as K
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        px, _ = self._process_input(images)
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            predictions = self._batch_pred_fn([px])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        input_shape = image.shape
        px, dpdx = self._process_input(image)
        predictions, gradient = self._pred_grad_fn([
            px[np.newaxis],
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        return predictions, gradient

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        px, dpdx = self._process_input(image)
        gradient = self._bw_grad_fn([
            gradient,
            px[np.newaxis],
        ])
        gradient = gradient[0]   # output of bw_grad_fn is a list
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == image.shape
        return gradient

class KerasModel2(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Keras` model.
    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel2, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        from tensorflow.compat.v1.keras import backend as K
        import tensorflow.compat.v1.keras as keras
        #from keras import backend as K
        #import keras
        from pkg_resources import parse_version

        #assert parse_version(keras.__version__) >= parse_version('2.0.7'), 'Keras version needs to be 2.0.7 or newer'  # noqa: E501

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        if predicts == 'probabilities':
            if False:
                pass
            # if K.backend() == 'tensorflow':
            #     predictions, = predictions.op.inputs
            #     loss = K.sparse_categorical_crossentropy(
            #         label_input, predictions, from_logits=True)
            else:
                logging.warning('relying on numerically unstable conversion'
                                ' from probabilities to softmax')
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=False)

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)
        grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            #pass
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            grad_loss_input = K.reshape(grad_loss_input, (1,) + grad_loss_input.shape)  # noqa: E501
        else:
            assert not isinstance(grads, list)
            grad = grads

            grad_loss_input = grads_loss_input

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        self._pred_grad_fn = K.function(
           [images_input, label_input],
           [predictions, grad])
        self._bw_grad_fn = K.function(
           [grad_loss_output, images_input],
           [grad_loss_input])

    def _to_logits(self, predictions):
        from tensorflow.compat.v1.keras import backend as K
        #from keras import backend as K
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        px, _ = self._process_input(images)
        predictions = self._batch_pred_fn([px])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        input_shape = image.shape
        px, dpdx = self._process_input(image)
        predictions, gradient = self._pred_grad_fn([
            px[np.newaxis],
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        return predictions, gradient

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        px, dpdx = self._process_input(image)
        gradient = self._bw_grad_fn([
            gradient,
            px[np.newaxis],
        ])
        gradient = gradient[0]   # output of bw_grad_fn is a list
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == image.shape
        return gradient

class KerasModel2Compat(DifferentiableModel):
    """Creates a :class:`Model` instance from a `Keras` model.
    Parameters
    ----------
    model : `keras.models.Model`
        The `Keras` model that should be attacked.
    bounds : tuple
        Tuple of lower and upper bound for the pixel values, usually
        (0, 1) or (0, 255).
    channel_axis : int
        The index of the axis that represents color channels.
    preprocessing: 2-element tuple with floats or numpy arrays
        Elementwises preprocessing of input; we first subtract the first
        element of preprocessing from the input and then divide the input by
        the second element.
    predicts : str
        Specifies whether the `Keras` model predicts logits or probabilities.
        Logits are preferred, but probabilities are the default.
    """

    def __init__(
            self,
            model,
            bounds,
            channel_axis=3,
            preprocessing=(0, 1),
            predicts='probabilities'):

        super(KerasModel2Compat, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        #from tensorflow.keras import backend as K
        #import tensorflow.keras as keras
        #from tensorflow.compat.v1.keras import backend as K
        #import tensorflow.compat.v1.keras as keras
        from keras import backend as K
        import keras
        from pkg_resources import parse_version

        #assert parse_version(keras.__version__) >= parse_version('2.0.7'), 'Keras version needs to be 2.0.7 or newer'  # noqa: E501

        if predicts == 'probs':
            predicts = 'probabilities'
        assert predicts in ['probabilities', 'logits']

        images_input = model.input
        label_input = K.placeholder(shape=(1,))

        predictions = model.output

        shape = K.int_shape(predictions)
        _, num_classes = shape
        assert num_classes is not None

        self._num_classes = num_classes

        if predicts == 'probabilities':
            if False:
                pass
            # if K.backend() == 'tensorflow':
            #     predictions, = predictions.op.inputs
            #     loss = K.sparse_categorical_crossentropy(
            #         label_input, predictions, from_logits=True)
            else:
                logging.warning('relying on numerically unstable conversion'
                                ' from probabilities to softmax')
                loss = K.sparse_categorical_crossentropy(
                    label_input, predictions, from_logits=False)

                # transform the probability predictions into logits, so that
                # the rest of this code can assume predictions to be logits
                predictions = self._to_logits(predictions)
        elif predicts == 'logits':
            loss = K.sparse_categorical_crossentropy(
                label_input, predictions, from_logits=True)

        # sparse_categorical_crossentropy returns 1-dim tensor,
        # gradients wants 0-dim tensor (for some backends)
        loss = K.squeeze(loss, axis=0)
        #grads = K.gradients(loss, images_input)

        grad_loss_output = K.placeholder(shape=(num_classes, 1))
        external_loss = K.dot(predictions, grad_loss_output)
        # remove batch dimension of predictions
        external_loss = K.squeeze(external_loss, axis=0)
        # remove singleton dimension of grad_loss_output
        external_loss = K.squeeze(external_loss, axis=0)

        #grads_loss_input = K.gradients(external_loss, images_input)

        if K.backend() == 'tensorflow':
            # tensorflow backend returns a list with the gradient
            # as the only element, even if loss is a single scalar
            # tensor;
            # theano always returns the gradient itself (and requires
            # that loss is a single scalar tensor)
            #assert isinstance(grads, list)
            #assert len(grads) == 1
            #grad = grads[0]

            #assert isinstance(grads_loss_input, list)
            #assert len(grads_loss_input) == 1
            #grad_loss_input = grads_loss_input[0]
            pass
        elif K.backend() == 'cntk':  # pragma: no cover
            assert isinstance(grads, list)
            assert len(grads) == 1
            grad = grads[0]
            grad = K.reshape(grad, (1,) + grad.shape)

            assert isinstance(grads_loss_input, list)
            assert len(grads_loss_input) == 1
            grad_loss_input = grads_loss_input[0]
            grad_loss_input = K.reshape(grad_loss_input, (1,) + grad_loss_input.shape)  # noqa: E501
        else:
            assert not isinstance(grads, list)
            grad = grads

            grad_loss_input = grads_loss_input

        self._loss_fn = K.function(
            [images_input, label_input],
            [loss])
        self._batch_pred_fn = K.function(
            [images_input], [predictions])
        #self._pred_grad_fn = K.function(
        #    [images_input, label_input],
        #    [predictions, grad])
        #self._bw_grad_fn = K.function(
        #    [grad_loss_output, images_input],
        #    [grad_loss_input])

    def _to_logits(self, predictions):
        #from tensorflow.keras import backend as K
        #from tensorflow.compat.v1.keras import backend as K
        from keras import backend as K
        eps = 10e-8
        predictions = K.clip(predictions, eps, 1 - eps)
        predictions = K.log(predictions)
        return predictions

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        px, _ = self._process_input(images)
        predictions = self._batch_pred_fn([px])
        assert len(predictions) == 1
        predictions = predictions[0]
        assert predictions.shape == (images.shape[0], self.num_classes())
        return predictions

    def predictions_and_gradient(self, image, label):
        input_shape = image.shape
        px, dpdx = self._process_input(image)
        predictions, gradient = self._pred_grad_fn([
            px[np.newaxis],
            np.array([label])])
        predictions = np.squeeze(predictions, axis=0)
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert predictions.shape == (self.num_classes(),)
        assert gradient.shape == input_shape
        return predictions, gradient

    def backward(self, gradient, image):
        assert gradient.ndim == 1
        gradient = np.reshape(gradient, (-1, 1))
        px, dpdx = self._process_input(image)
        gradient = self._bw_grad_fn([
            gradient,
            px[np.newaxis],
        ])
        gradient = gradient[0]   # output of bw_grad_fn is a list
        gradient = np.squeeze(gradient, axis=0)
        gradient = self._process_gradient(dpdx, gradient)
        assert gradient.shape == image.shape
        return gradient
abstractmethod = abc.abstractmethod

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta('ABC', (), {})

    
class Criterion(ABC):
    """Base class for criteria that define what is adversarial.
    The :class:`Criterion` class represents a criterion used to
    determine if predictions for an image are adversarial given
    a reference label. It should be subclassed when implementing
    new criteria. Subclasses must implement is_adversarial.
    """

    def name(self):
        """Returns a human readable name that uniquely identifies
        the criterion with its hyperparameters.
        Returns
        -------
        str
            Human readable name that uniquely identifies the criterion
            with its hyperparameters.
        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.
        """
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, predictions, label):
        """Decides if predictions for an image are adversarial given
        a reference label.
        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.
        Returns
        -------
        bool
            True if an image with the given predictions is an adversarial
            example when the ground-truth class is given by label, False
            otherwise.
        """
        raise NotImplementedError

    def __and__(self, other):
        return CombinedCriteria(self, other)


class Misclassification(Criterion):
    """Defines adversarials as images for which the predicted class
    is not the original class.
    See Also
    --------
    :class:`TopKMisclassification`
    Notes
    -----
    Uses `numpy.argmax` to break ties.
    """

    def name(self):
        return 'Top1Misclassification'

    def is_adversarial(self, predictions, label):
        top1 = np.argmax(predictions)
        return top1 != label


@functools.total_ordering
class Distance(ABC):
    """Base class for distances.
    This class should be subclassed when implementing
    new distances. Subclasses must implement _calculate.
    """

    def __init__(
            self,
            reference=None,
            other=None,
            bounds=None,
            value=None):

        if value is not None:
            # alternative constructor
            assert isinstance(value, Number)
            assert reference is None
            assert other is None
            assert bounds is None
            self.reference = None
            self.other = None
            self._bounds = None
            self._value = value
            self._gradient = None
        else:
            # standard constructor
            self.reference = reference
            self.other = other
            self._bounds = bounds
            self._value, self._gradient = self._calculate()

        assert self._value is not None

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @abstractmethod
    def _calculate(self):
        """Returns distance and gradient of distance w.r.t. to self.other"""
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '{} = {:.6e}'.format(self.name(), self._value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value == other.value

    def __lt__(self, other):
        if other.__class__ != self.__class__:
            raise TypeError('Comparisons are only possible between the same distance types.')  # noqa: E501
        return self.value < other.value


class MeanSquaredDistance(Distance):
    """Calculates the mean squared error between two images.
    """

    def _calculate(self):
        min_, max_ = self._bounds
        n = self.reference.size
        f = n * (max_ - min_)**2

        diff = self.other - self.reference
        value = np.vdot(diff, diff) / f

        # calculate the gradient only when needed
        self._g_diff = diff
        self._g_f = f
        gradient = None
        return value, gradient

    @property
    def gradient(self):
        if self._gradient is None:
            self._gradient = self._g_diff / (self._g_f / 2)
        return self._gradient

    def __str__(self):
        return 'normalized MSE = {:.2e}'.format(self._value)


MSE = MeanSquaredDistance

class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""
    pass


class Adversarial(object):
    """Defines an adversarial that should be found and stores the result.
    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference image. It can be passed to
    an adversarial attack to find the actual adversarial.
    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which images are adversarial.
    original_image : a :class:`numpy.ndarray`
        The original image to which the adversarial image should
        be as close as possible.
    original_class : int
        The ground-truth label of the original image.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between images.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.
    """
    def __init__(
            self,
            model,
            criterion,
            original_image,
            original_class,
            distance=MSE,
            threshold=None,
            verbose=False):

        self.__model = model
        self.__criterion = criterion
        self.__original_image = original_image
        self.__original_image_for_distance = original_image
        self.__original_class = original_class
        self.__distance = distance

        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self.__threshold = threshold

        self.verbose = verbose

        self.__best_adversarial = None
        self.__best_distance = distance(value=np.inf)
        self.__best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        # check if the original image is already adversarial
        try:
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                self.predictions(original_image)
        except StopAttack:
            # if a threshold is specified and the original input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.

    def _reset(self):
        self.__best_adversarial = None
        self.__best_distance = self.__distance(value=np.inf)
        self.__best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self.predictions(self.__original_image)

    @property
    def image(self):
        """The best adversarial found so far."""
        return self.__best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.
        None if no adversarial has been found.
        """
        return self.__best_adversarial_output

    @property
    def adversarial_class(self):
        """The argmax of the model predictions for the best adversarial found so far.
        None if no adversarial has been found.
        """
        if self.output is None:
            return None
        return np.argmax(self.output)

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self.__best_distance

    @property
    def original_image(self):
        """The original input."""
        return self.__original_image

    @property
    def original_class(self):
        """The class of the original input (ground-truth, not model prediction)."""  # noqa: E501
        return self.__original_class

    @property
    def _model(self):  # pragma: no cover
        """Should not be used."""
        return self.__model

    @property
    def _criterion(self):  # pragma: no cover
        """Should not be used."""
        return self.__criterion

    @property
    def _distance(self):  # pragma: no cover
        """Should not be used."""
        return self.__distance

    def set_distance_dtype(self, dtype):
        assert dtype >= self.__original_image.dtype
        self.__original_image_for_distance = self.__original_image.astype(
            dtype, copy=False)

    def reset_distance_dtype(self):
        self.__original_image_for_distance = self.__original_image

    def normalized_distance(self, image):
        """Calculates the distance of a given image to the
        original image.
        Parameters
        ----------
        image : `numpy.ndarray`
            The image that should be compared to the original image.
        Returns
        -------
        :class:`Distance`
            The distance between the given image and the original image.
        """
        return self.__distance(
            self.__original_image_for_distance,
            image,
            bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""
        return self.__threshold is not None \
            and self.__best_distance <= self.__threshold

    def __new_adversarial(self, image, predictions, in_bounds):
        image = image.copy()  # to prevent accidental inplace changes
        distance = self.normalized_distance(image)
        if in_bounds and self.__best_distance > distance:
            # new best adversarial
            if self.verbose:
                print('new best adversarial: {}'.format(distance))

            self.__best_adversarial = image
            self.__best_distance = distance
            self.__best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def __is_adversarial(self, image, predictions, in_bounds):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.
        Parameters
        ----------
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some image.
        label : int
            The label of the unperturbed reference image.
        """
        is_adversarial = self.__criterion.is_adversarial(
            predictions, self.__original_class)
        assert isinstance(is_adversarial, bool) or \
            isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self.__new_adversarial(
                image, predictions, in_bounds)
            if self.verbose:
                print('found adversarial: {}'.format(self.normalized_distance(image)))
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance

    def target_class(self):
        """Interface to criterion.target_class for attacks.
        """
        try:
            target_class = self.__criterion.target_class()
        except AttributeError:
            target_class = None
        return target_class

    def num_classes(self):
        n = self.__model.num_classes()
        assert isinstance(n, numbers.Number)
        return n

    def bounds(self):
        min_, max_ = self.__model.bounds()
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, batch):
        """Interface to model.channel_axis for attacks.
        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of images
            (4 dimensions) or a single image (3 dimensions) should be returned.
        """
        axis = self.__model.channel_axis()
        if not batch:
            axis = axis - 1
        return axis

    def has_gradient(self):
        """Returns true if _backward and _forward_backward can be called
        by an attack, False otherwise.
        """
        try:
            self.__model.gradient
            self.__model.predictions_and_gradient
        except AttributeError:
            return False
        else:
            return True

    def predictions(self, image, strict=True, return_details=False):
        """Interface to model.predictions for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            predictions = self.__model.predictions(image)
        is_adversarial, is_best, distance = self.__is_adversarial(
            image, predictions, in_bounds)
        assert predictions.ndim == 1
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

    def batch_predictions(
            self, images, greedy=False, strict=True, return_details=False):
        """Interface to model.batch_predictions for attacks.
        Parameters
        ----------
        images : `numpy.ndarray`
            Batch of inputs with shape as expected by the model.
        greedy : bool
            Whether the first adversarial should be returned.
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        if strict:
            in_bounds = self.in_bounds(images)
            assert in_bounds

        self._total_prediction_calls += len(images)
        predictions = self.__model.batch_predictions(images)

        assert predictions.ndim == 2
        assert predictions.shape[0] == images.shape[0]

        if return_details:
            assert greedy

        adversarials = []
        for i in range(len(predictions)):
            if strict:
                in_bounds_i = True
            else:
                in_bounds_i = self.in_bounds(images[i])
            is_adversarial, is_best, distance = self.__is_adversarial(
                images[i], predictions[i], in_bounds_i)
            if is_adversarial and greedy:
                if return_details:
                    return predictions, is_adversarial, i, is_best, distance
                else:
                    return predictions, is_adversarial, i
            adversarials.append(is_adversarial)

        if greedy:  # pragma: no cover
            # no adversarial found
            if return_details:
                return predictions, False, None, False, None
            else:
                return predictions, False, None

        is_adversarial = np.array(adversarials)
        assert is_adversarial.ndim == 1
        assert is_adversarial.shape[0] == images.shape[0]

        return predictions, is_adversarial

    def gradient(self, image=None, label=None, strict=True):
        """Interface to model.gradient for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        assert self.has_gradient()

        if image is None:
            image = self.__original_image
        if label is None:
            label = self.__original_class

        assert not strict or self.in_bounds(image)

        self._total_gradient_calls += 1
        gradient = self.__model.gradient(image, label)

        assert gradient.shape == image.shape
        return gradient

    def predictions_and_gradient(
            self, image=None, label=None, strict=True, return_details=False):
        """Interface to model.predictions_and_gradient for attacks.
        Parameters
        ----------
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
            Defaults to the original image.
        label : int
            Label used to calculate the loss that is differentiated.
            Defaults to the original label.
        strict : bool
            Controls if the bounds for the pixel values should be checked.
        """
        assert self.has_gradient()

        if image is None:
            image = self.__original_image
        if label is None:
            label = self.__original_class

        in_bounds = self.in_bounds(image)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        self._total_gradient_calls += 1
        predictions, gradient = self.__model.predictions_and_gradient(image, label)  # noqa: E501
        is_adversarial, is_best, distance = self.__is_adversarial(
            image, predictions, in_bounds)

        assert predictions.ndim == 1
        assert gradient.shape == image.shape
        if return_details:
            return predictions, gradient, is_adversarial, is_best, distance
        else:
            return predictions, gradient, is_adversarial

    def backward(self, gradient, image=None, strict=True):
        """Interface to model.backward for attacks.
        Parameters
        ----------
        gradient : `numpy.ndarray`
            Gradient of some loss w.r.t. the logits.
        image : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        Returns
        -------
        gradient : `numpy.ndarray`
            The gradient w.r.t the image.
        See Also
        --------
        :meth:`gradient`
        """
        assert self.has_gradient()
        assert gradient.ndim == 1

        if image is None:
            image = self.__original_image

        assert not strict or self.in_bounds(image)

        self._total_gradient_calls += 1
        gradient = self.__model.backward(gradient, image)

        assert gradient.shape == image.shape
        return gradient


class Attack(ABC):
    """Abstract base class for adversarial attacks.
    The :class:`Attack` class represents an adversarial attack that searches
    for adversarial examples. It should be subclassed when implementing new
    attacks.
    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    criterion : a :class:`Criterion` instance
        The criterion that determines which images are adversarial.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    distance : a :class:`Distance` class
        The measure used to quantify similarity between images.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.
        Ignored if the attack is called with an :class:`Adversarial` instance.
    Notes
    -----
    If a subclass overwrites the constructor, it should call the super
    constructor with *args and **kwargs.
    """

    def __init__(self,
                 model=None, criterion=Misclassification(),
                 distance=MSE, threshold=None):
        self._default_model = model
        self._default_criterion = criterion
        self._default_distance = distance
        self._default_threshold = threshold

        # to customize the initialization in subclasses, please
        # try to overwrite _initialize instead of __init__ if
        # possible
        self._initialize()

    def _initialize(self):
        """Additional initializer that can be overwritten by
        subclasses without redefining the full __init__ method
        including all arguments and documentation."""
        pass

    @abstractmethod
    def __call__(self, input_or_adv, label=None, unpack=True, **kwargs):
        raise NotImplementedError

    def name(self):
        """Returns a human readable name that uniquely identifies
        the attack with its hyperparameters.
        Returns
        -------
        str
            Human readable name that uniquely identifies the attack
            with its hyperparameters.
        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.
        """
        return self.__class__.__name__


def call_decorator(call_fn):
    @functools.wraps(call_fn)
    def wrapper(self, input_or_adv, label=None, unpack=True, **kwargs):
        assert input_or_adv is not None

        if isinstance(input_or_adv, Adversarial):
            a = input_or_adv
            if label is not None:
                raise ValueError('Label must not be passed when input_or_adv'
                                 ' is an Adversarial instance')
        else:
            if label is None:
                raise ValueError('Label must be passed when input_or_adv is'
                                 ' not an Adversarial instance')
            else:
                model = self._default_model
                criterion = self._default_criterion
                distance = self._default_distance
                threshold = self._default_threshold
                if model is None or criterion is None:
                    raise ValueError('The attack needs to be initialized'
                                     ' with a model and a criterion or it'
                                     ' needs to be called with an Adversarial'
                                     ' instance.')
                a = Adversarial(model, criterion, input_or_adv, label,
                                distance=distance, threshold=threshold)

        assert a is not None

        if a.distance.value == 0.:
            warnings.warn('Not running the attack because the original input'
                          ' is already misclassified and the adversarial thus'
                          ' has a distance of 0.')
        elif a.reached_threshold():
            warnings.warn('Not running the attack because the given treshold'
                          ' is already reached')
        else:
            try:
                _ = call_fn(self, a, label=None, unpack=None, **kwargs)
                assert _ is None, 'decorated __call__ method must return None'
            except StopAttack:
                # if a threshold is specified, StopAttack will be thrown
                # when the treshold is reached; thus we can do early
                # stopping of the attack
                logging.info('threshold reached, stopping attack')

        if a.image is None:
            warnings.warn('{} did not find an adversarial, maybe the model'
                          ' or the criterion is not supported by this'
                          ' attack.'.format(self.name()))

        if unpack:
            return a.image
        else:
            return a

    return wrapper
class SingleStepGradientBaseAttack(Attack):
    """Common base class for single step gradient attacks."""

    @abc.abstractmethod
    def _gradient(self, a):
        raise NotImplementedError

    def _run(self, a, epsilons, max_epsilon):
        if not a.has_gradient():
            return

        image = a.original_image
        min_, max_ = a.bounds()

        gradient = self._gradient(a)

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, max_epsilon, num=epsilons + 1)[1:]
            decrease_if_first = True
        else:
            decrease_if_first = False

        for _ in range(2):  # to repeat with decreased epsilons if necessary
            for i, epsilon in enumerate(epsilons):
                perturbed = image + gradient * epsilon
                perturbed = np.clip(perturbed, min_, max_)

                _, is_adversarial = a.predictions(perturbed)                
                if is_adversarial:
                    if decrease_if_first and i < 20:
                        print('repeating attack with smaller epsilons')
                        break
                    return

            max_epsilon = epsilons[i]
            epsilons = np.linspace(0, max_epsilon, num=20 + 1)[1:]


class GradientSignAttack(SingleStepGradientBaseAttack):
    """Adds the sign of the gradient to the image, gradually increasing
    the magnitude until the image is misclassified. This attack is
    often referred to as Fast Gradient Sign Method and was introduced
    in [1]_.
    Does not do anything if the model does not have a gradient.
    References
    ----------
    .. [1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy,
           "Explaining and Harnessing Adversarial Examples",
           https://arxiv.org/abs/1412.6572
    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=1000, max_epsilon=1):

        """Adds the sign of the gradient to the image, gradually increasing
        the magnitude until the image is misclassified.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of step sizes in the direction of the sign of
            the gradient or number of step sizes between 0 and max_epsilon
            that should be tried.
        max_epsilon : float
            Largest step size if epsilons is not an iterable.
        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        return self._run(a, epsilons=epsilons, max_epsilon=max_epsilon)

    def _gradient(self, a):
        min_, max_ = a.bounds()
        gradient = a.gradient()
        gradient = np.sign(gradient) * (max_ - min_)
        return gradient


FGSM = GradientSignAttack

