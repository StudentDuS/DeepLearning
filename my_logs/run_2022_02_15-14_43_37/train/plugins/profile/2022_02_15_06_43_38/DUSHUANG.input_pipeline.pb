	??%䃞????%䃞??!??%䃞??	1vIj?
,@1vIj?
,@!1vIj?
,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??%䃞????A?f??Am???{???Y?N@aã?*	     ?T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatY?8??m??!?>Q=h?G@)V-???1???L?]A@:Preprocessing2F
Iterator::Model?(??0??!A:?2	v=@)Dio??ɔ?1?-Nk?O8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor46<?R??!'??*@)46<?R??1'??*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatelxz?,C??!J?????0@)"??u????11 K?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_?Lu?!_n??@)??_?Lu?1_n??@:Preprocessing2U
Iterator::Model::ParallelMapV2"??u??q?!1 K?@)"??u??q?11 K?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipGx$(??!pqZ?}?Q@)???_vOn?1e?*|?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapK?=?U??!sCH?R2@)?~j?t?X?1?2	v???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 14.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t15.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.92vIj?
,@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??A?f????A?f??!??A?f??      ??!       "      ??!       *      ??!       2	m???{???m???{???!m???{???:      ??!       B      ??!       J	?N@aã??N@aã?!?N@aã?R      ??!       Z	?N@aã??N@aã?!?N@aã?JCPU_ONLYY2vIj?
,@b 