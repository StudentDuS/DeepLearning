"?=
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff?d?@9ffff?d?@Affff?d?@Iffff?d?@a?H?????i?H??????Unknown?
BHostIDLE"IDLE1?????T?@A?????T?@a??????i??????Unknown
dHostDataset"Iterator::Model(1     ?@@9     ?@@A?????L<@I?????L<@aL?fՆS?i?	?}R????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1333333'@9333333'@A333333'@I333333'@aM0 ??@?i?I??R????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333*@9333333*@A??????%@I??????%@ac?#r??=?io??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??????%@9??????%@A??????%@I??????%@ac?#r??=?i??`??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff'@9ffffff'@A??????"@I??????"@a?haP??9?i?J?????Unknown
iHostWriteSummary"WriteSummary(1??????@9??????@A??????@I??????@a=?????5?i??۔?????Unknown?
t	Host_FusedMatMul"sequential/dense_1/BiasAdd(1333333@9333333@A333333@I333333@a??X??~*?i?łf????Unknown
s
HostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@a?haP??)?i#???????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a?U?|??&?i????o????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a#(?.d&?i{r5??????Unknown
gHostStridedSlice"strided_slice(1333333@9333333@A333333@I333333@aU????%?i?|?n)????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @aC˨?Q#?i?	??^????Unknown
^HostGatherV2"GatherV2(1??????	@9??????	@A??????	@I??????	@a??????!?i??*y????Unknown
VHostSum"Sum_2(1??????	@9??????	@A??????	@I??????	@a??????!?ig?5ɓ????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@ac?#r???i?w	??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff<@9ffffff<@A      @I      @a,?F:}??i?I?
_????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a,?F:}??i???;????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a?:j@d?iF/??????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff??Affffff@Iffffff??a?:j@d?i?B??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1??????@9??????@A??????@I??????@a[?{f?I?iwv?g?????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????? @9?????? @A?????? @I?????? @a????/?i????M????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1       @9       @A       @I       @a#(?.d?i?? ??????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a#(?.d?iմA&?????Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff??9ffffff??Affffff??Iffffff??a?̰????i[Jn?V????Unknown
eHost
LogicalAnd"
LogicalAnd(1333333??9333333??A333333??I333333??aQ?Z???i? ??????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1333333??9333333??A333333??I333333??aQ?Z???i???<?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a^?"K??iXM?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff??9ffffff??Affffff??Iffffff??a?Y??i?G?\?????Unknown
VHostMean"Mean(1????????9????????A????????I????????a?M5???iu? -?????Unknown
v HostAssignAddVariableOp"AssignAddVariableOp_2(1333333??9333333??A333333??I333333??a??X??~
?i??(`????Unknown
s!HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333??9333333??A333333??I333333??a??X??~
?i9?$?????Unknown
?"HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1333333??9333333??A333333??I333333??a??X??~
?i??4????Unknown
?#HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff,@9ffffff,@A????????I????????a[?{f?I?i??F?????Unknown
p$HostSquaredDifference"SquaredDifference(1????????9????????A????????I????????a[?{f?I?iy@?l?????Unknown
b%HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????a[?{f?I?ih? ?W????Unknown
X&HostCast"Cast_1(1      ??9      ??A      ??I      ??a#(?.d?i唱??????Unknown
}'HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      ??9      ??A      ??I      ??a#(?.d?ibOB6????Unknown
u(HostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a#(?.d?i?	Ӈ`????Unknown
})HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a#(?.d?i\?cٸ????Unknown
V*HostCast"Cast(1????????9????????A????????I????????a???????i??
??????Unknown
+HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????a???????i???(F????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>i??c??????Unknown
|-HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>i????????Unknown
w.HostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>i?Ƞ?????Unknown
w/HostCast"%gradient_tape/mean_squared_error/Cast(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>i,zs=????Unknown
u0HostSub"$gradient_tape/mean_squared_error/sub(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>i>H,F{????Unknown
|1HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a?Y??>ibd??????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1333333??9333333??A333333??I333333??a??X??~?>i???????Unknown
T3HostMul"Mul(1333333??9333333??A333333??I333333??a??X??~?>i??X#????Unknown
`4HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??a??X??~?>iuX????Unknown
u5HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a#(?.d?>i?w?:?????Unknown
y6HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a#(?.d?>i?Ԧc?????Unknown
?7HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a#(?.d?>i/2o??????Unknown
w8HostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a#(?.d?>im?7?????Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a???????>i8	,????Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a???????>i??\O????Unknown
};HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a???????>i???r????Unknown
u<HostMul"$gradient_tape/mean_squared_error/Mul(1333333??9333333??A333333??I333333??a??X??~?>i'??/?????Unknown
?=HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??a??X??~?>i?Eo??????Unknown
?>HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a??X??~?>i??M-?????Unknown
??HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??a??X??~?>i2?,??????Unknown
a@HostIdentity"Identity(1????????9????????A????????I????????a???????>iAV?????Unknown?
?AHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a???????>i?????????Unknown*?=
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff?d?@9ffff?d?@Affff?d?@Iffff?d?@a???????i????????Unknown?
dHostDataset"Iterator::Model(1     ?@@9     ?@@A?????L<@I?????L<@aV?ҧT?i)??????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1333333'@9333333'@A333333'@I333333'@agl_?mu@?iv8????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1333333*@9333333*@A??????%@I??????%@a?<b{??>?iL?M??????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1??????%@9??????%@A??????%@I??????%@a?<b{??>?i?N}??????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff'@9ffffff'@A??????"@I??????"@aF+????:?i???9????Unknown
iHostWriteSummary"WriteSummary(1??????@9??????@A??????@I??????@aj????j6?i?t???????Unknown?
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1333333@9333333@A333333@I333333@aRRW?=+?i_?x?????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1??????@9??????@A??????@I??????@aF+????*?ib"?BJ????Unknown
?
HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a?@uO?D'?i?L??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??音&?i??J?)????Unknown
gHostStridedSlice"strided_slice(1333333@9333333@A333333@I333333@a??ƃU"&?i????????Unknown
?HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1      @9      @A      @I      @a?Vj?+?#?i??a??????Unknown
^HostGatherV2"GatherV2(1??????	@9??????	@A??????	@I??????	@a????L)"?i?d-X?????Unknown
VHostSum"Sum_2(1??????	@9??????	@A??????	@I??????	@a????L)"?iH??????Unknown
`HostGatherV2"
GatherV2_1(1??????@9??????@A??????@I??????@a?<b{???iZ??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff<@9ffffff<@A      @I      @ak???`?i???????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ak???`?i?.#?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a9?L^?i?????????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff??Affffff@Iffffff??a9?L^?iD???k????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1??????@9??????@A??????@I??????@a"??????i4?3????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????? @9?????? @A?????? @I?????? @a	hL?4??i}??P?????Unknown
?HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1       @9       @A       @I       @a??音?in????????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1       @9       @A       @I       @a??音?i_E؊]????Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff??9ffffff??Affffff??Iffffff??a?????i?41
????Unknown
eHost
LogicalAnd"
LogicalAnd(1333333??9333333??A333333??I333333??a?/???K?ivi=r?????Unknown?
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1333333??9333333??A333333??I333333??a?/???K?i?I?>????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??at?6???i?	?????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1ffffff??9ffffff??Affffff??Iffffff??a??GF??i4"(F????Unknown
VHostMean"Mean(1????????9????????A????????I????????a???i???4?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1333333??9333333??A333333??I333333??aRRW?=?i3Ta,)????Unknown
s HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333??9333333??A333333??I333333??aRRW?=?i??-$?????Unknown
?!HostDivNoNan"?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nan(1333333??9333333??A333333??I333333??aRRW?=?i??????Unknown
?"HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff,@9ffffff,@A????????I????????a"??????i? ?f????Unknown
p#HostSquaredDifference"SquaredDifference(1????????9????????A????????I????????a"??????i?F??????Unknown
b$HostDivNoNan"div_no_nan_1(1????????9????????A????????I????????a"??????i?"l?.????Unknown
X%HostCast"Cast_1(1      ??9      ??A      ??I      ??a??音?i&?듉????Unknown
}&HostMaximum"(gradient_tape/mean_squared_error/Maximum(1      ??9      ??A      ??I      ??a??音?i?okb?????Unknown
u'HostSum"$gradient_tape/mean_squared_error/Sum(1      ??9      ??A      ??I      ??a??音?i?0?????Unknown
}(HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      ??9      ??A      ??I      ??a??音?i??j??????Unknown
V)HostCast"Cast(1????????9????????A????????I????????a????L)?i"????????Unknown
*HostFloorDiv")gradient_tape/mean_squared_error/floordiv(1????????9????????A????????I????????a????L)?i???I+????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i?!]?j????Unknown
|,HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i???j?????Unknown
w-HostReadVariableOp"div_no_nan/ReadVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i>v??????Unknown
w.HostCast"%gradient_tape/mean_squared_error/Cast(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i:??)????Unknown
u/HostSub"$gradient_tape/mean_squared_error/sub(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i[Z?i????Unknown
|0HostDivNoNan"&mean_squared_error/weighted_loss/value(1ffffff??9ffffff??Affffff??Iffffff??a??GF??>i|???????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_4(1333333??9333333??A333333??I333333??aRRW?=?>i+)?????Unknown
T2HostMul"Mul(1333333??9333333??A333333??I333333??aRRW?=?>i?I??????Unknown
`3HostDivNoNan"
div_no_nan(1333333??9333333??A333333??I333333??aRRW?=?>i?z? L????Unknown
u4HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??音?>i?M?y????Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??音?>i!N??????Unknown
?6HostBroadcastTo",gradient_tape/mean_squared_error/BroadcastTo(1      ??9      ??A      ??I      ??a??音?>i=??V?????Unknown
w7HostMul"&gradient_tape/mean_squared_error/mul_1(1      ??9      ??A      ??I      ??a??音?>iy?ͽ????Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_1(1????????9????????A????????I????????a????L)?>iC=g&????Unknown
w9HostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????a????L)?>i? cJ????Unknown
}:HostRealDiv"(gradient_tape/mean_squared_error/truediv(1????????9????????A????????I????????a????L)?>i?(??n????Unknown
u;HostMul"$gradient_tape/mean_squared_error/Mul(1333333??9333333??A333333??I333333??aRRW?=?>i.A???????Unknown
?<HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??aRRW?=?>i?Y?1?????Unknown
?=HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??aRRW?=?>i?qso?????Unknown
?>HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??aRRW?=?>i3?f??????Unknown
a?HostIdentity"Identity(1????????9????????A????????I????????a????L)?>iE???????Unknown?
?@HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1????????9????????A????????I????????a????L)?>i?????????Unknown