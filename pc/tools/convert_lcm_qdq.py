"""Convert LCM-Dreamshaper INT8 UNet from QOperator (ConvInteger) to QDQ.

ORT Android supports QDQ (QuantizeLinear/DequantizeLinear) but not ConvInteger.
The rewrite keeps int8 weights and inserts DequantizeLinear nodes; ORT fuses
them into quantized kernels.

Pattern per ConvInteger:
  sample -> DynamicQuantizeLinear -> (x_q, x_scale, x_zp)
  ConvInteger(x_q, w_q, x_zp, w_zp) -> int32
  Cast(int32->float) -> Mul(scale) -> Add(bias) -> out
Rewrite to:
  DequantizeLinear(x_q, x_scale, x_zp) -> x_f
  DequantizeLinear(w_q, w_scale, w_zp) -> w_f
  Conv(x_f, w_f, bias) -> out
"""
from pathlib import Path
import onnx
from onnx import helper, numpy_helper

SRC = Path(r"D:\AI\DiffusionServer\models\LCM_Dreamshaper_ONNX\unet\model.onnx")
DST = Path(r"D:\AI\DiffusionServer\models\LCM_Dreamshaper_ONNX\unet\model_qdq.onnx")

model = onnx.load(str(SRC))
g = model.graph
producer = {o: n for n in g.node for o in n.output}
consumer = {}
for n in g.node:
    for i in n.input:
        consumer.setdefault(i, []).append(n)
inits = {i.name: i for i in g.initializer}

removed = set()
new_nodes = []
new_inits = []

count = 0
skipped = 0
for n in list(g.node):
    if n.op_type != "ConvInteger":
        continue
    x_q, w_q, x_zp, w_zp = n.input
    dq = producer.get(x_q)
    if dq is None or dq.op_type != "DynamicQuantizeLinear":
        skipped += 1
        continue
    x_float = dq.input[0]
    x_scale = dq.output[1]
    x_zp_out = dq.output[2]

    if w_q not in inits or w_zp not in inits:
        skipped += 1
        continue

    prefix = w_q[:-len(".weight_quantized")] if w_q.endswith(".weight_quantized") else w_q
    w_scale_name = prefix + ".weight_scale"
    if w_scale_name not in inits:
        skipped += 1
        continue

    # downstream chain: Cast -> Mul -> (Add bias)
    O = n.output[0]
    cast = next((c for c in consumer.get(O, []) if c.op_type == "Cast"), None)
    if cast is None:
        skipped += 1
        continue
    C = cast.output[0]
    mul = next((c for c in consumer.get(C, []) if c.op_type == "Mul"), None)
    if mul is None:
        skipped += 1
        continue
    M = mul.output[0]
    add = next((c for c in consumer.get(M, []) if c.op_type == "Add"), None)
    if add is not None:
        F = add.output[0]
        bias = add.input[1] if add.input[0] == M else add.input[0]
        if bias not in inits:
            bias = None
    else:
        F = M
        bias = None

    x_deq_out = x_q + "_deq"
    w_deq_out = w_q + "_deq"

    new_nodes.append(helper.make_node(
        "DequantizeLinear", [x_q, x_scale, x_zp_out], [x_deq_out], name=x_q + "_dq"
    ))
    new_nodes.append(helper.make_node(
        "DequantizeLinear", [w_q, w_scale_name, w_zp], [w_deq_out], name=w_q + "_dq"
    ))

    conv_inputs = [x_deq_out, w_deq_out] + ([bias] if bias else [])
    attrs = {a.name: a for a in n.attribute}
    kwargs = {}
    for a in n.attribute:
        if a.name == "kernel_shape":
            kwargs["kernel_shape"] = list(a.ints)
        elif a.name == "strides":
            kwargs["strides"] = list(a.ints)
        elif a.name == "pads":
            kwargs["pads"] = list(a.ints)
        elif a.name == "dilations":
            kwargs["dilations"] = list(a.ints)
        elif a.name == "group":
            kwargs["group"] = a.i
        elif a.name == "auto_pad":
            kwargs["auto_pad"] = a.s.decode()
    new_nodes.append(helper.make_node(
        "Conv", conv_inputs, [F], name=n.name + "_conv", **kwargs
    ))
    removed.add(n.name)
    removed.add(cast.name)
    removed.add(mul.name)
    if add is not None:
        removed.add(add.name)
    count += 1

# rebuild node list
kept = [n for n in g.node if n.name not in removed] + new_nodes

# drop unused DynamicQuantizeLinear nodes
used = set()
for n in kept:
    used.update(n.input)
final_nodes = []
for n in kept:
    if n.op_type == "DynamicQuantizeLinear" and not any(o in used for o in n.output):
        continue
    final_nodes.append(n)

del g.node[:]
g.node.extend(final_nodes)

# drop unused initializers
used = set()
for n in g.node:
    used.update(n.input)
for o in g.output:
    used.add(o.name)
keep_inits = [i for i in g.initializer if i.name in used]
del g.initializer[:]
g.initializer.extend(keep_inits)

print(f"converted={count} skipped={skipped} nodes={len(g.node)} inits={len(g.initializer)}")

# Save with external data (weights stay int8)
onnx.save_model(
    model, str(DST),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=DST.name + ".data",
    size_threshold=1024,
)
print("saved:", DST, "size MB:", DST.stat().st_size / 1048576)
data = Path(str(DST) + ".data")
print("data MB:", data.stat().st_size / 1048576)
