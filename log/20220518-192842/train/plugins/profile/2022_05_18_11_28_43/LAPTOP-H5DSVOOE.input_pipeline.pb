	?V?/?'???V?/?'??!?V?/?'??	t(??	@t(??	@!t(??	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V?/?'??6<?R?!??AI??&??Y$????ۧ?*	?????S@2F
Iterator::Model???B?i??!?????|C@)???????1?#??9r>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?z6?>??!???(??=@)Έ?????1?F??h8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatee?X???!-AK??6@)?]K?=??1?C??=t1@:Preprocessing2U
Iterator::Model::ParallelMapV29??v??z?!D??=t!@)9??v??z?1D??=t!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+??Χ?!a0?N@)HP?s?r?1	Z???%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!a[???@)	?^)?p?1a[???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!@?O?S?@)????Mbp?1@?O?S?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*???!L&??d29@)ŏ1w-!_?1?(?ʏ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9t(??	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6<?R?!??6<?R?!??!6<?R?!??      ??!       "      ??!       *      ??!       2	I??&??I??&??!I??&??:      ??!       B      ??!       J	$????ۧ?$????ۧ?!$????ۧ?R      ??!       Z	$????ۧ?$????ۧ?!$????ۧ?JCPU_ONLYYt(??	@b 