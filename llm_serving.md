# LLM服务化部署

## 部署架构
- 负载均衡：Nginx/HAProxy
- 模型服务：vLLM/TensorRT-LLM
- API网关：限流、鉴权
- 监控：Prometheus+Grafana

## vLLM部署
```python
from vllm import LLM
llm = LLM(model="llama-2-7b", 
          tensor_parallel_size=4,
          max_num_seqs=256)
outputs = llm.generate(prompts, sampling_params)
```

## 性能优化
- Continuous Batching
- PagedAttention
- 量化推理INT8/INT4
- KV cache复用

## 成本控制
- Spot实例
- 自动扩缩容
- 模型缓存
- 请求队列管理
