	???JY??????JY???!???JY???	?]?$?#&@?]?$?#&@!?]?$?#&@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???JY???????Mb??A?1w-!??Y?V-??*????̌M@)       =2F
Iterator::Model??+e???!#????D@)??~j?t??16?&)@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatjM????!"?]??(@@)?o_???1?VH?A<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM??St$??!i??0?3@)?5?;Nс?1??^Kq-@:Preprocessing2U
Iterator::Model::ParallelMapV2?g??s?u?!?W?-??!@)?g??s?u?1?W?-??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?5?;Nѡ?!??^KqM@)F%u?k?1t{-9?U@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Le?!Џ-??@)??_?Le?1Џ-??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!q?)`>@)a2U0*?c?1q?)`>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!?\?5<?4@)/n??R?1??<L????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t39.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?]?$?#&@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Mb??????Mb??!????Mb??      ??!       "      ??!       *      ??!       2	?1w-!???1w-!??!?1w-!??:      ??!       B      ??!       J	?V-???V-??!?V-??R      ??!       Z	?V-???V-??!?V-??JCPU_ONLYY?]?$?#&@b 