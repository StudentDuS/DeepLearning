?	?D???6@?D???6@!?D???6@	?O6?)7@?O6?)7@!?O6?)7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?D???6@?S㥛%@A?,C?"@Y?k	??'@*	??????@2F
Iterator::Model-!?l@!?g????X@) o??@1a??X??X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Pk?w??!???;???)???????1?H
??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateA??ǘ???!?$?????)?St$????1?y8[?	??:Preprocessing2U
Iterator::Model::ParallelMapV29??v??z?!p&@b??)9??v??z?1p&@b??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?W[?????!:̧:??)y?&1?l?1????"???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceǺ???f?!?????	??)Ǻ???f?1?????	??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!?0T??5??)HP?s?b?1?0T??5??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C??6??!@Ï????)_?Q?[?1??_??j??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 23.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t46.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?O6?)7@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S㥛%@?S㥛%@!?S㥛%@      ??!       "      ??!       *      ??!       2	?,C?"@?,C?"@!?,C?"@:      ??!       B      ??!       J	?k	??'@?k	??'@!?k	??'@R      ??!       Z	?k	??'@?k	??'@!?k	??'@JCPU_ONLYY?O6?)7@b Y      Y@qu*W?T@"?	
host?Your program is HIGHLY input-bound because 23.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t46.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?83.2709% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 