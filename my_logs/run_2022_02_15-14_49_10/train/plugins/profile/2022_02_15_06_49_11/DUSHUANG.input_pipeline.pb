	6<?R???6<?R???!6<?R???	?ð?hB)@?ð?hB)@!?ð?hB)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$6<?R????^)???A|a2U0??Y䃞ͪϥ?*	hffff?a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???o_??!???8P@)?e??a???1d&SIjO@:Preprocessing2F
Iterator::Model??j+????!?q??G?4@)??ǘ????1O3?'@:Preprocessing2U
Iterator::Model::ParallelMapV2?
F%u??! ???v"@)?
F%u??1 ???v"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ?o_Ή?!v??!@)n????1?b?V?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW[??잼?!??n?S@)?HP?x?1?x?F?W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!ל^.} @)?????g?1ל^.} @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!aph>???)Ǻ???f?1aph>???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H?}??!c?l?x$@)??H?}]?1c?l?x??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t26.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?ð?hB)@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^)????^)???!?^)???      ??!       "      ??!       *      ??!       2	|a2U0??|a2U0??!|a2U0??:      ??!       B      ??!       J	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?R      ??!       Z	䃞ͪϥ?䃞ͪϥ?!䃞ͪϥ?JCPU_ONLYY?ð?hB)@b 