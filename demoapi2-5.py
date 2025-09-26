import os
import logging
import time
import requests
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置和初始化
load_dotenv()
app = Flask(__name__)
CORS(app)

# 检索配置
SEARCH_API_URL = "http://222.29.51.95:4042/search"
SIMILARITY_API_URL = "http://222.29.51.95:4042/similarity"
MAX_WORKERS_GENERATION = 1
GENERATION_COUNT = 1
MAX_RESULTS = 5
MAX_WORKERS_SIMILARITY = 50

# ARK API配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL_ID = "deepseek-v3-1-250821"

# 日志配置
logging.basicConfig(
    filename="demoapi2-5.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger(__name__)

def segment_text_intelligently(text: str, min_chunk_len: int = 150, max_chunk_len: int = 500):
    """智能分割文本为指定长度的块"""
    step_start = time.time()
    initial_chunks = [line.strip() for line in text.split('\n') if line.strip()]
    merged_chunks = []
    current = ""

    for i, chunk in enumerate(initial_chunks):
        if not current:
            current = chunk
        else:
            potential_len = len(current) + len(chunk) + 1
            should_merge = (len(current) < min_chunk_len and i < len(initial_chunks) - 1) or \
                          (potential_len <= max_chunk_len * 1.1 and len(current) < max_chunk_len)
            if should_merge:
                current += "\n" + chunk
            else:
                merged_chunks.append(current)
                current = chunk
    if current:
        merged_chunks.append(current)

    final_segments = []
    for segment in merged_chunks:
        if len(segment) > max_chunk_len:
            splits = [s.strip() for s in re.split(r'(?<=[。？！])', segment) if s.strip()]
            sub_current = ""
            for j, sent in enumerate(splits):
                if not sub_current:
                    sub_current = sent
                else:
                    potential_sub = len(sub_current) + len(sent)
                    if (potential_sub <= max_chunk_len * 1.1) and \
                       (len(sub_current) < min_chunk_len or j == len(splits) - 1):
                        sub_current += sent
                    else:
                        final_segments.append(sub_current)
                        sub_current = sent
            if sub_current:
                final_segments.append(sub_current)
        else:
            final_segments.append(segment)

    result = [s.strip() for s in final_segments if s.strip()]
    step_time = round(time.time() - step_start, 4)
    logger.info(f"文本分割完成 - 原始长度: {len(text)}, 分割后块数: {len(result)}, 耗时: {step_time}秒")
    return result

def calculate_similarity(sent1: str, sent2: str) -> float:
    """计算两个文本的相似度"""
    step_start = time.time()
    if not sent1 or not sent2:
        logger.info(f"相似度计算 - 输入文本为空, 耗时: {round(time.time() - step_start, 4)}秒")
        return 0.0
        
    try:
        response = requests.post(
            SIMILARITY_API_URL,
            json={"sent1": sent1, "sent2": sent2},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        score = float(response.json().get("score", 0.0)) if response.status_code == 200 else 0.0
        step_time = round(time.time() - step_start, 4)
        logger.info(f"相似度计算完成 - 分数: {score}, 耗时: {step_time}秒")
        return score
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"相似度计算错误: {str(e)}, 耗时: {step_time}秒")
        return 0.0

class ArkApiAdapter:
    """Ark API适配器"""
    def __init__(self, api_key, base_url, model):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "你是人工智能助手") -> str:
        """调用Ark API生成内容"""
        step_start = time.time()
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content
            step_time = round(time.time() - step_start, 4)
            logger.info(f"Ark API调用完成 - 生成内容长度: {len(result)}, 耗时: {step_time}秒")
            return result
        except Exception as e:
            step_time = round(time.time() - step_start, 4)
            logger.error(f"Ark API调用错误: {e}, 耗时: {step_time}秒")
            return f"Error from API: {str(e)}"

# 初始化LLM客户端
llm = ArkApiAdapter(ARK_API_KEY, ARK_BASE_URL, ARK_MODEL_ID) if ARK_API_KEY else None
if not llm:
    logger.warning("未配置ARK_API_KEY，大模型增强功能不可用")

def generate_related_materials(text: str, generation_num: int, function_type: str) -> str:
    """生成补充材料"""
    step_start = time.time()
    if not text or not llm:
        step_time = round(time.time() - step_start, 4)
        logger.info(f"补充材料生成 #{generation_num+1} - 跳过(无文本或LLM), 耗时: {step_time}秒")
        return ""
        
    system_prompt = """你是专业内容验证助手，基于原文本生成一段补充材料：
    1. 围绕原文本核心内容展开，补充相关背景信息
    2. 使用客观、规范的表述，符合百科内容风格
    3. 包含2-3个与核心内容相关的细节描述
    4. 长度50-80字，仅返回补充材料本身，不添加任何解释说明""" if function_type == "verify" else \
    """你是专业问答辅助助手，基于问题生成一段补充材料：
    1. 围绕问题核心展开，补充相关背景信息
    2. 使用客观、规范的表述，符合百科内容风格
    3. 包含2-3个与问题相关的细节描述，帮助定位答案
    4. 长度50-80字，仅返回补充材料本身，不添加任何解释说明"""
        
    try:
        result = llm.generate(f"根据以下{ '文本' if function_type == 'verify' else '问题' }生成补充材料：\n\n{text}", system_prompt).strip()
        step_time = round(time.time() - step_start, 4)
        logger.info(f"补充材料生成 #{generation_num+1} - 长度: {len(result)}, 耗时: {step_time}秒, 内容: {result[:50]}...")
        return result
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"补充材料生成错误 #{generation_num+1}: {str(e)}, 耗时: {step_time}秒")
        return ""

def parse_verify_result(llm_result):
    """解析校验结果"""
    step_start = time.time()
    parsed = {"corrected_text": "", "explanation": ""}
    if not llm_result:
        step_time = round(time.time() - step_start, 4)
        logger.info(f"校验结果解析 - 输入为空, 耗时: {step_time}秒")
        return parsed
    
    try:
        if "修正文本：" in llm_result:
            match = re.search(r'修正文本：(.*?)(?=修正说明：|$)', llm_result, re.DOTALL)
            parsed["corrected_text"] = match.group(1).strip() if match else ""
        else:
            end_pattern = re.compile(r'修正说明：|解释：|说明：', re.IGNORECASE)
            match = end_pattern.search(llm_result)
            parsed["corrected_text"] = (llm_result[:match.start()].strip() if match else llm_result.strip())
        
        if "修正说明：" in llm_result:
            match = re.search(r'修正说明：(.*)', llm_result, re.DOTALL)
            parsed["explanation"] = match.group(1).strip() if match else ""
            
        parsed["corrected_text"] = re.sub(r'^修正文本：\s*', '', parsed["corrected_text"], flags=re.IGNORECASE)
        step_time = round(time.time() - step_start, 4)
        logger.info(f"校验结果解析完成 - 修正文本长度: {len(parsed['corrected_text'])}, 说明长度: {len(parsed['explanation'])}, 耗时: {step_time}秒")
        return parsed
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"解析校验结果错误: {str(e)}, 耗时: {step_time}秒")
        return parsed

def find_closest_reference(target_text, target_supplements, references):
    """查找最相似的参考材料（使用问题+补充材料联合匹配）"""
    step_start = time.time()
    if not target_text or not references:
        step_time = round(time.time() - step_start, 4)
        logger.info(f"查找最相似材料 - 无目标文本或参考材料, 耗时: {step_time}秒")
        return None
        
    try:
        # 合并目标文本和补充材料作为联合匹配依据
        combined_target_parts = [target_text.strip()]
        if target_supplements:
            combined_target_parts.extend([s.strip() for s in target_supplements if s.strip()])
        combined_target = "\n\n".join(combined_target_parts)
        target_clean = re.sub(r'\s+', ' ', combined_target)
        
        max_sim, closest = 0.0, None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SIMILARITY) as executor:
            # 用联合目标与每个参考材料计算相似度
            futures = {
                executor.submit(calculate_similarity, target_clean, re.sub(r'\s+', ' ', item.get('content', '').strip())): item 
                for item in references
            }
            
            for future in as_completed(futures):
                item = futures[future]
                try:
                    sim = future.result()
                    item["score"] = sim
                    if sim > max_sim:
                        max_sim, closest = sim, item
                except Exception as e:
                    logger.error(f"处理相似度结果错误: {str(e)}")
                    item["score"] = 0.0

        result = closest if closest else {"content": "无匹配材料", "doctitle": "匹配结果", "score": 0.0}
        step_time = round(time.time() - step_start, 4)
        logger.info(f"查找最相似材料完成 - 最高相似度: {max_sim}, 来源: {result.get('doctitle')}, 联合目标长度: {len(combined_target)}, 耗时: {step_time}秒")
        return result
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"查找最相似材料错误: {str(e)}, 耗时: {step_time}秒")
        return None

def find_most_relevant_content(text, results, function_type, supplements=None):
    """找到最相关的内容（使用问题+补充材料联合匹配）"""
    step_start = time.time()
    if not text or not results or not llm:
        step_time = round(time.time() - step_start, 4)
        logger.info(f"内容相关性分析 - 无足够输入, 耗时: {step_time}秒")
        return {"processed_content": text, "corrected_text": text, "explanation": "", "closest_reference": None}
        
    try:
        system_prompt = """你是一个专业的内容校验专家，擅长基于提供的所有基准材料验证并更正文本。
        请仔细分析待验证文本和所有提供的基准材料（包括补充材料），根据所有材料的内容，对需要验证的文本进行全面更正。
        修正后的结果需要与对应的材料完全一致，包括事实、数据和表述。
        必须按照以下格式输出，不允许添加任何其他内容：
        修正文本：[在此处填写修正后的完整文本]
        修正说明：
        [错误类型1]“原文内容1”——>“修正后内容1”
        [错误类型2]“原文内容2”——>“修正后内容2”
        [错误类型3]“原文内容3”——>“修正后内容3”
        ...
        （每个修正项单独成行）""" if function_type == "verify" else \
        """你是一个专业的问答专家，擅长基于所有提供的基准材料回答问题。
        请仔细分析问题和所有提供的基准材料（包括补充材料），按以下步骤进行处理：
        1. 基于所有材料的综合信息提供两部分回答：
           a. 简洁回答：根据待查询问题和所有材料，总结出最精炼的回答（不超过30个字）
           b. 详细回答：基于所有材料的综合信息，提供完整、详细的回答
           两部分回答用"简洁回答："和"详细回答："作为开头明确区分，且两部分之间必须空一行
        2. 如果没有找到相关的材料或材料不足以回答问题，请直接返回"无法回答"
        
        核心要求：
        - 所有回答必须完全基于提供的所有材料，不得添加外部知识
        - 绝对不提及任何关于材料、信息来源或材料的存在
        - 不要包含任何其他标题、说明或解释性文字"""

        # 准备参考材料（仅包含词条，补充材料不加入词条列表）
        refs = [f"{item.get('doctitle', '无标题')}：{item['content']}" for item in results]
        # 补充材料作为独立参考信息提供给模型
        if supplements:
            for i, m in enumerate(supplements):
                refs.append(f"补充材料_{i+1}：{m}")
        refs_text = "\n\n".join(refs)

        prompt = f"""待验证文本：{text}
        
        所有参考材料（包括基准材料和补充材料）：
        {refs_text}
        
        请基于上述所有材料进行综合校验和更正，严格按照系统提示中的格式返回结果，不要包含任何额外信息。""" if function_type == "verify" else \
        f"""待查询问题：{text}
        
        所有参考材料（包括基准材料和补充材料）：
        {refs_text}
        
        请基于上述所有材料进行综合回答，严格按照系统提示中的要求和格式返回结果，不要包含任何额外信息。"""

        result = llm.generate(prompt, system_prompt).strip()
        
        # 使用问题+补充材料联合匹配最相关的词条
        if function_type == "verify":
            parsed = parse_verify_result(result)
            # 用修正后的文本+补充材料联合匹配
            combined_target = [parsed["corrected_text"]] + (supplements if supplements else [])
            closest_ref = find_closest_reference(parsed["corrected_text"], supplements, results)
            step_time = round(time.time() - step_start, 4)
            logger.info(f"内容校验完成 - 修正文本长度: {len(parsed['corrected_text'])}, 参考材料数: {len(results)}, 补充材料数: {len(supplements) if supplements else 0}, 耗时: {step_time}秒")
            return {
                "processed_content": result,
                "corrected_text": parsed["corrected_text"],
                "explanation": parsed["explanation"],
                "closest_reference": closest_ref
            }
        else:
            # 用原始问题+补充材料联合匹配
            closest_ref = find_closest_reference(text, supplements, results)
            step_time = round(time.time() - step_start, 4)
            logger.info(f"问答处理完成 - 回答长度: {len(result)}, 参考材料数: {len(results)}, 补充材料数: {len(supplements) if supplements else 0}, 耗时: {step_time}秒")
            return {
                "processed_content": result,
                "corrected_text": "",
                "explanation": "",
                "closest_reference": closest_ref
            }
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"内容处理错误: {str(e)}, 耗时: {step_time}秒")
        return {"processed_content": text, "corrected_text": text, "explanation": f"处理错误: {str(e)}", "closest_reference": None}

def external_api_search(text: str, method_name: str) -> dict:
    """调用外部检索API"""
    step_start = time.time()
    try:
        response = requests.post(SEARCH_API_URL, json={"text": text}, timeout=30)
        if response.status_code != 200:
            step_time = round(time.time() - step_start, 4)
            logger.error(f"检索API失败: {response.status_code}, 耗时: {step_time}秒")
            return {"items": [], "combined_content": ""}
        
        items = []
        for item in response.json().get("result", []):
            items.append({
                "content": item.get("content", ""),
                "doctitle": item.get("word", ""),
                "source_method": method_name
            })
        step_time = round(time.time() - step_start, 4)
        logger.info(f"检索完成 - 结果数: {len(items)}, 查询长度: {len(text)}, 耗时: {step_time}秒")
        return {
            "items": items,
            "combined_content": "\n\n".join([i["content"] for i in items if i["content"]])
        }
    except Exception as e:
        step_time = round(time.time() - step_start, 4)
        logger.error(f"检索错误: {str(e)}, 耗时: {step_time}秒")
        return {"items": [], "combined_content": ""}

def run_combined_search(text, use_llm, function_type):
    """执行组合检索（含文本分割和补充材料生成）"""
    step_start = time.time()
    processed_text = text
    combined_texts = [processed_text]
    generated = []
    gen_info = {"generated_materials": [], "total_generated": 0, "valid_generated": 0, "generation_errors": 0}

    if use_llm and llm:
        errors = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_GENERATION) as executor:
            futures = [executor.submit(generate_related_materials, processed_text, i, function_type) for i in range(GENERATION_COUNT)]
            for i, future in enumerate(as_completed(futures)):
                try:
                    mat = future.result()
                    if mat and len(mat.strip()) > 10:
                        generated.append(mat)
                except:
                    errors += 1
        
        gen_info = {
            "generated_materials": generated,
            "total_generated": GENERATION_COUNT,
            "valid_generated": len(generated),
            "generation_errors": errors
        }
        combined_texts.extend(generated)

    result = external_api_search("\n\n".join(combined_texts), 
                                f"combined_search_with_llm_{function_type}" if use_llm else f"basic_combined_search_{function_type}")

    # 文本分割处理
    segmented = []
    for item in result["items"]:
        for seg in segment_text_intelligently(item["content"], 400, 500):
            new_item = item.copy()
            new_item["content"] = seg
            segmented.append(new_item)

    step_time = round(time.time() - step_start, 4)
    logger.info(f"组合检索流程完成 - 原始检索结果数: {len(result['items'])}, 分割后结果数: {len(segmented)}, 补充材料数: {len(generated)}, 耗时: {step_time}秒")
    return {
        "items": segmented,
        "combined_content": "\n\n".join([i["content"] for i in segmented if i["content"]]),
        "generation_info": gen_info,
        "final_query": "\n\n".join(combined_texts),
        "supplementary_materials": generated
    }

@app.route('/search_references', methods=['POST'])
def search_references_endpoint():
    """API端点：检索参考材料"""
    overall_start = time.time()
    logger.info("="*50)
    logger.info("新的API请求开始处理")
    
    try:
        # 解析请求参数
        parse_start = time.time()
        data = request.get_json()
        if not data or not data.get("text_to_search"):
            return jsonify({"error": "缺少 'text_to_search' 参数"}), 400
        
        text = data.get("text_to_search")
        use_llm = data.get("use_llm_enhancement", False)
        function_type = data.get("function_type")
        
        if function_type not in ["verify", "qa"]:
            return jsonify({"error": "无效的 'function_type' 参数"}), 400
        
        if use_llm and not llm:
            logger.warning("LLM不可用，禁用增强功能")
            use_llm = False
        
        parse_time = round(time.time() - parse_start, 4)
        logger.info(f"请求参数解析完成 - 文本长度: {len(text)}, 功能类型: {function_type}, LLM增强: {use_llm}, 耗时: {parse_time}秒")

        # 执行检索
        search_result = run_combined_search(text, use_llm, function_type)
        all_items = search_result["items"]  # 仅包含分割后的词条，不包含补充材料
        gen_info = search_result["generation_info"]
        final_query = search_result["final_query"]
        supplements = search_result["supplementary_materials"]

        # 准备匹配依据：问题+补充材料的组合
        sim_start = time.time()
        combined_target_parts = [text.strip()]
        if supplements:
            combined_target_parts.extend([s.strip() for s in supplements if s.strip()])
        combined_target = "\n\n".join(combined_target_parts)
        text_clean = re.sub(r'\s+', ' ', combined_target)
        
        # 仅对分割后的词条计算相似度（使用问题+补充材料联合匹配）
        sim_scores = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_SIMILARITY) as executor:
            futures = {
                executor.submit(calculate_similarity, text_clean, re.sub(r'\s+', ' ', item.get('content', '').strip())): item 
                for item in all_items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    sim = future.result()
                    item["score"] = round(sim, 4)
                    sim_scores.append((item, sim))
                except:
                    item["score"] = 0.0
                    sim_scores.append((item, 0.0))

        sim_time = round(time.time() - sim_start, 4)
        logger.info(f"相似度计算完成 - 词条数: {len(all_items)}, 联合匹配文本长度: {len(combined_target)}, 耗时: {sim_time}秒")

        # 排序和限制结果（仅包含词条）
        sort_start = time.time()
        sorted_results = [item for item, _ in sorted(sim_scores, key=lambda x: x[1], reverse=True)]
        limited = sorted_results[:MAX_RESULTS]
        sort_time = round(time.time() - sort_start, 4)
        logger.info(f"结果排序完成 - 排序前: {len(sorted_results)}, 限制后: {len(limited)}, 耗时: {sort_time}秒")

        # 处理内容（使用问题+补充材料联合匹配）
        processed = find_most_relevant_content(text, limited, function_type, supplements)

        # 准备响应（返回参数不变，仅包含词条）
        response_start = time.time()
        for item in limited:
            item.pop("source_method", None)

        response = {
            "status": "success",
            "processed_content": processed["corrected_text"] if function_type == "verify" else processed["processed_content"],
            "reference_material": "\n\n".join([i["content"] for i in limited if i["content"]]),
            "items": limited,  # 仅包含分割后的词条
            "processing_time_seconds": round(time.time() - overall_start, 4),
            "result_count": len(limited),
            "use_llm_enhancement": use_llm,
            "function_type": function_type,
            "supplementary_materials_used": len(supplements) > 0,
            "intermediate_results": {
                "generation_info": gen_info,
                "final_combined_query": final_query,
                "deduplication_stats": {"before": len(sorted_results), "after": len(sorted_results), "removed": 0}
            }
        }

        if function_type == "verify":
            response["verify_components"] = {"explanation": processed["explanation"]}
        response["closest_reference"] = processed["closest_reference"]

        response_time = round(time.time() - response_start, 4)
        overall_time = round(time.time() - overall_start, 4)
        logger.info(f"响应准备完成 - 响应大小: {len(str(response))}, 耗时: {response_time}秒")
        logger.info(f"API请求处理完成 - 总耗时: {overall_time}秒")
        logger.info("="*50)

        return jsonify(response), 200

    except Exception as e:
        overall_time = round(time.time() - overall_start, 4)
        logger.error(f"API错误: {e}, 总耗时: {overall_time}秒")
        logger.info("="*50)
        return jsonify({"error": "检索处理失败", "details": str(e)}), 500

if __name__ == '__main__':
    logger.info("服务初始化成功，开始运行")
    app.run(host='0.0.0.0', port=8081, debug=True)