from __future__ import print_function

"""
Extra set of callbacks.

Author: Mateusz Malinowski
Email: mmalinow@mpi-inf.mpg.de
"""


import random
import warnings
import numpy as np

#from bokeh.plotting import cursession
#from bokeh.plotting import figure
#from bokeh.plotting import push
#from bokeh.plotting import show

from keras.callbacks import Callback as KerasCallback

#from agnez.keras_callbacks import BokehCallback

from ..utils import print_metrics
from ..utils.read_write import dict2file 
from ..utils.read_write import print_qa


def is_era_begin(epoch, epoch_interval):
    return (epoch+1) % epoch_interval == 0 or epoch == 0


def is_era_end(epoch, epoch_interval):
    return (epoch+1) % epoch_interval == 0 


###
# Storing callbacks
###
class StoreModelWeightsOnEraEnd(KerasCallback):
    def __init__(self, filepath, epoch_interval, verbose=0):
        """
        In:
            filepath - formattable filepath; possibilities:
                * weights.{epoch:02d}
                * weights.{era:02d}
            epoch_interval - 
                number of epochs that must be passed from the previous saving
            verbose - if nonzero then print out information on stdout;
                by default 0
        """
        super(KerasCallback, self).__init__()
        self.filepath = filepath
        self.epoch_interval = epoch_interval
        self.verbose = verbose
        self.era = 0

    def on_epoch_end(self, epoch, logs={}):
        if is_era_end(epoch, self.epoch_interval):
            filepath = self.filepath.format(
                    epoch=epoch, era=self.era, **logs)
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)
            self.era += 1
###

###
# Printing callbacks
###
class PrintOnEraBegin(KerasCallback):
    def __init__(self, epoch_interval, message='Era {era:02d}'):
        """
        In:
            epoch_interval - 
                number of epochs that must be passed between two consecutive
                invocations of this callback
            message -
                formattable message to show;
                by default "Era {era:02d}" showing the current era
        """
        self.epoch_interval = epoch_interval
        self.era = 0
        self.message = message

    def on_epoch_begin(self, epoch, logs={}):
        if is_era_begin(epoch, self.epoch_interval):
            print()
            print('-' * 50)
            print(self.message.format(era=self.era))
            self.era += 1


class PrintPerformanceMetricOnEraEnd(KerasCallback):
    def __init__(self, X, y, temperature, index2word_y, 
            metric_name, epoch_interval, extra_vars, 
            verbosity_path='logs/performance.log', verbose=1):
        """
        In:
            X - encoded input
            y - raw expected output
            temperature - temperature for the predictions;
                the colder the temperature the more stable answers
            index2word_y - mapping from the indices to words (in the y-domain)
            metric_name - name of the performance metric
            epoch_interval - 
                number of epochs that must be passed between two consecutive
                invocations of this callback
            extra_vars - dictionary of extra variables
            verbosity path - path to dumb the logs
            verbose - verbosity level;
                by default 1
        """
        self.X = X
        self.y = y
        self.temperature = temperature
        self.index2word_y = index2word_y
        self.metric_name = metric_name
        self.epoch_interval = epoch_interval
        self.extra_vars = extra_vars
        self.verbosity_path = verbosity_path
        self.verbose = verbose
        self.era = 0

    def on_epoch_end(self, epoch, logs={}):
        if is_era_end(epoch, self.epoch_interval):
            answer_pred = self.model.decode_predictions(
                    X=self.X, 
                    temperature=self.temperature, 
                    index2word=self.index2word_y, 
                    verbose=self.verbose)
            metric_values = print_metrics.select[self.metric_name](
                    gt_list=self.y, 
                    pred_list=answer_pred, 
                    verbose=1,
                    extra_vars=self.extra_vars)
            if self.verbose == 1:
                for m in metric_values:
                    if 'idiosyncrasy' in m:
                        idiosyncrasies = m['idiosyncrasy'].split(':')
                        if 'long' in idiosyncrasies and 'muted' in idiosyncrasies:
                            # long value being muted, we can only send the results
                            # to the file
                            filepath = self.verbosity_path.format(
                                    epoch=epoch, era=self.era, **logs)
                            if m['value'] is not None:
                                dict2file(m['value'], filepath, title=m['name'])
            self.era += 1
###

###
# Plotting callbacks
###
'''
class PlotPerformanceMetricOnEraEnd(BokehCallback):
    """
    Plots the performance measures.

    Inspired by 
        https://github.com/EderSantana/agnez/blob/master/agnez/keras_callbacks.py
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, X, y, temperature, index2word_y, 
            metric_name, epoch_interval, extra_vars, verbose=1, 
            name='experiment', fig_title='Performance', url='http://127.0.0.1:5006'):
        """
        In:
            X - encoded input
            y - raw expected output
            temperature - temperature for the predictions;
                the colder the temperature the more stable answers
            index2word_y - mapping from the indices to words (in the y-domain)
            metric_name - name of the performance metric
            epoch_interval - 
                number of epochs that must be passed between two consecutive
                invocations of this callback
            extra_vars - dictionary of extra variables
            verbose - verbosity level; by default 1
            name - name of the bokeh document; by default 'experiment'
            fig_title - title of the bokeh figure; by default 'Performance'
            url - bokeh server url; 
                by default 'http://127.0.0.1:5006'
        """
        BokehCallback.__init__(self, name, fig_title, url)
        self.X = X
        self.y = y
        self.temperature = temperature
        self.index2word_y = index2word_y
        self.metric_name = metric_name
        self.epoch_interval = epoch_interval
        self.extra_vars = extra_vars
        self.verbose = verbose
        self.era = 0

    def on_epoch_end(self, epoch, logs={}):
        if not is_era_end(epoch, self.epoch_interval):
            return

        answer_pred = self.model.decode_predictions(
                    X=self.X, 
                    temperature=self.temperature, 
                    index2word=self.index2word_y, 
                    verbose=self.verbose)
        measures = print_metrics.select[self.metric_name](
                gt_list=self.y, 
                pred_list=answer_pred, 
                verbose=1,
                extra_vars=self.extra_vars)

        if not hasattr(self, 'fig'):
            self.fig = figure(title=self.fig_title)
            for i, m in enumerate(measures):
                if 'idiosyncrasy' in m:
                    if 'muted' in m['idiosyncrasy'].split(':'):
                        continue
                self.fig.line([self.era], [m['value']], legend=m['name'],
                              name=m['name'], line_width=2,
                              line_color=self.colors[i % len(self.colors)])
                renderer = self.fig.select({'name': m['name']})
                self.plots.append(renderer[0].data_source)
            show(self.fig)
        else:
            for i, m in enumerate(measures):
                if 'idiosyncrasy' in m:
                    if 'muted' in m['idiosyncrasy'].split(':'):
                        continue
                self.plots[i].data['y'].append(m['value'])
                self.plots[i].data['x'].append(self.era)
        cursession().store_objects(self.plots[i])
        push()
        self.era += 1

class StandardPerformancePlot(BokehCallback):
    """
    Generalizes Agnez class Plot to work with all standard performance metrics.

    Original work: Eder Santana [https://github.com/EderSantana]
    """
    # WIP
    # TODO:
    #   -[ ] Decide API for choosing channels to plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    """ 
    Inspired by https://github.com/mila-udem/blocks-extras/blob/master/blocks/extras/extensions/plot.py

    """ 
    def __init__(self, 
            what_to_plot=['loss', 'val_loss'], 
            name='experiment', 
            fig_title='Cost functions', 
            url='default'):
        BokehCallback.__init__(self, name, fig_title, url)
        self.totals = {}
        self.what_to_plot = what_to_plot

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        what_to_plot_now = list(set(['loss', 'acc']) & set(self.what_to_plot))
        for v in what_to_plot_now:
            if v in self.totals:
                self.totals[v] += logs.get(v) * batch_size
            else:
                self.totals[v] = logs.get(v) * batch_size

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self, 'fig'):
            self.fig = figure(title=self.fig_title)
            for i, v in enumerate(self.what_to_plot):
                if v == 'loss':
                    L = self.totals[v] / self.seen
                else:
                    L = logs.get(v)
                self.fig.line([epoch], [L], legend=v,
                              name=v, line_width=2,
                              line_color=self.colors[i % len(self.colors)])
                renderer = self.fig.select({'name': v})
                self.plots.append(renderer[0].data_source)
            show(self.fig)
        else:
            for i, v in enumerate(self.what_to_plot):
                if v in ['loss', 'acc']:
                    L = self.totals[v] / self.seen
                else:
                    L = logs.get(v)
                self.plots[i].data['y'].append(L)
                self.plots[i].data['x'].append(epoch)
        cursession().store_objects(self.plots[i])
        push()
###
'''

###
# Monitoring callbacks
###
class MonitorPredictionsOnEndEra(KerasCallback):
    """
    Checks the performance on a randomly chosen subset of the data.
    Hopefully the network generates something interesting.
    """
    def __init__(self, X, x, y, temperature, index2word_y,
            verbosity_path, epoch_interval, subset_size=0, verbose=0):
        """
        In:
            X - encoded input
            x - raw input
            y - raw output space
            temperature - temperature for the predictions;
                the colder the temperature the more stable answers
            index2word_y - mapping from the indices to words (in the y-domain)
            verbosity_path - parameterized filepath to store the logs;
                possibilities of the parameterization
                * weights.{epoch:02d}
                * weights.{era:02d}
            epoch_interval - 
                number of epochs that must be passed between two consecutive
                invocations of this callback
            subset_size - data subset's size, if 0 then all data are taken;
                by default 0
            verbose - verbosity level; by default 0
        """
        self.X = X
        self.x = x
        self.y = y
        self.temperature = temperature
        self.index2word_y = index2word_y
        self.verbosity_path = verbosity_path
        self.epoch_interval = epoch_interval
        self.subset_size = subset_size
        self.verbose = verbose
        self.era = 0

    def on_epoch_end(self, epoch, logs={}):
        if is_era_end(epoch, self.epoch_interval):
            if self.subset_size > 0:
                subset_indices = random.sample(
                        xrange(self.X.shape[0]), self.subset_size)
                if len(self.X) == 2:
                    X_subset = [self.X[0][subset_indices], self.X[1][subset_indices]]
                elif len(self.X) == 1:
                    X_subset = self.X[subset_indices] 
                questions_subset = self.x[subset_indices]
                answer_gt = self.y[subset_indices]
                answer_gt_original = self.y[subset_indices]
            else:
                X_subset = self.X
                questions_subset = self.x 
                answer_gt = self.y
                answer_gt_original = self.y 
            answer_pred = self.model.decode_predictions(
                    X=X_subset, 
                    temperature=self.temperature, 
                    index2word=self.index2word_y, 
                    verbose=self.verbose)

            filepath = self.verbosity_path.format(
                    epoch=epoch, era=self.era, **logs)
            print_qa(questions_subset, answer_gt, answer_gt_original, answer_pred, 
                    self.era, path=filepath)
            self.era += 1
###

###
# Learning modifiers callbacks
###
class LearningRateReducerWithEarlyStopping(KerasCallback):
    """
    Reduces learning rate during the training.

    Original work: jiumem [https://github.com/jiumem]
    """
    def __init__(self, 
            patience=0, reduce_rate=0.5, reduce_nb=10, 
            is_early_stopping=True, verbose=1):
        """
        In:
            patience - number of beginning epochs without reduction; 
                by default 0
            reduce_rate - multiplicative rate reducer; by default 0.5
            reduce_nb - maximal number of reductions performed; by default 10
            is_early_stopping - if true then early stopping is applied when
                reduce_nb is reached; by default True
            verbose - verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.is_early_stopping = is_early_stopping
        self.verbose = verbose
        self.epsilon = 0.1e-10

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score is None:
            warnings.warn('validation score is off; ' + 
                    'this reducer works only with the validation score on')
            return
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = np.float32(self.model.optimizer.lr.get_value())
                    self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                    if self.verbose > 0:
                        print("Reduction from {0:0.6f} to {1:0.6f}".\
                                format(float(lr), float(lr*self.reduce_rate)))
                    if float(lr) <= self.epsilon:
                        if self.verbose > 0:
                            print('Learning rate too small, learning stops now')
                        self.model.stop_training = True
                else:
                    if self.is_early_stopping:
                        if self.verbose > 0:
                            print("Epoch %d: early stopping" % (epoch))
                        self.model.stop_training = True
            self.wait += 1 


class LearningRateReducerEveryPatienceEpoch(KerasCallback):
    """
    Reduces learning rate during the training after every 'patience' epochs.

    Original work: jiumem [https://github.com/jiumem]
    """
    def __init__(self, 
            patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
        """
        In:
            patience - number of epochs in stagnation; by default 0
            reduce_rate - multiplicative rate reducer; by default 0.5
            reduce_nb - maximal number of reductions performed; by default 10
            verbose - verbosity level; by default 1
        """
        super(KerasCallback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.is_early_stopping = False
        self.verbose = verbose
        self.epsilon = 0.1e-10

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score is None:
            current_score = -10.0 # always reduce
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    lr = np.float32(self.model.optimizer.lr.get_value())
                    self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                    if self.verbose > 0:
                        print("Reduction from {0:0.6f} to {1:0.6f}".\
                                format(float(lr), float(lr*self.reduce_rate)))
                    if float(lr) <= self.epsilon:
                        if self.verbose > 0:
                            print('Learning rate too small, learning stops now')
                        self.model.stop_training = True
                else:
                    if self.is_early_stopping:
                        if self.verbose > 0:
                            print("Epoch %d: early stopping" % (epoch))
                        self.model.stop_training = True
                self.wait = 0
            else:
                self.wait += 1 
