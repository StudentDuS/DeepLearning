?	r?鷯??r?鷯??!r?鷯??	??z?Q$@??z?Q$@!??z?Q$@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$r?鷯??????????A&S????Y      ??*	fffffFU@2F
Iterator::Model8??d?`??!?Ru^bG@)??ܵ?|??1?n0E>?B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ?????!u??lR:@)???&??1??o?a?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?St$????!䎧?3@)lxz?,C??1v?)?Y70@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!?Ć???!@)ŏ1w-!?1?Ć???!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo?ŏ1??!୊??J@)U???N@s?1?l-?v@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vOn?!???,d@)???_vOn?1???,d@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!u??lR
@)Ǻ???f?1u??lR
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!_?ѽ3<@)-C??6Z?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t32.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??z?Q$@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	&S????&S????!&S????:      ??!       B      ??!       J	      ??      ??!      ??R      ??!       Z	      ??      ??!      ??JCPU_ONLYY??z?Q$@b Y      Y@qIMVR8?X@"?	
both?Your program is MODERATELY input-bound because 10.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t32.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?99.0347% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 