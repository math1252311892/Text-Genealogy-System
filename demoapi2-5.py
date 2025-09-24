import os
import logging
import time
import sys
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# 配置和初始化
load_dotenv()
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 检索配置
SEARCH_API_URL = "http://222.29.51.95:4042/search"
MAX_WORKERS_GENERATION = 1
GENERATION_COUNT = 1
MAX_RESULTS = 3

# ARK API配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL_ID = "deepseek-v3-1-250821"
EXTERNAL_SCORE_SCALE = (0, 100)

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
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger(__name__)

# Ark API适配器类
class ArkApiAdapter:
    def __init__(self, api_key, base_url, model):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "你是人工智能助手") -> str:
        """封装对Ark API的调用"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Ark API: {e}", exc_info=True)
            return f"Error from API: {str(e)}"

# 初始化LLM客户端
llm = None
try:
    if ARK_API_KEY:
        llm = ArkApiAdapter(ARK_API_KEY, ARK_BASE_URL, ARK_MODEL_ID)
        logger.info("LLM客户端初始化成功")
    else:
        logger.warning("未配置ARK_API_KEY，大模型增强功能将不可用")
except Exception as e:
    logger.error(f"LLM客户端初始化失败: {str(e)}", exc_info=True)

def generate_related_materials(text: str, generation_num: int, function_type: str) -> str:
    """生成便于百科检索的补充材料，专注于辅助检索词条内容"""
    if not text or not llm:
        return ""
        
    start = time.time()
    try:
        # 根据功能类型设置不同的系统提示，生成简洁补充材料
        if function_type == "verify":
            system_prompt = """你是专业内容验证助手，基于原文本生成一段补充材料：
            1. 围绕原文本核心内容展开，补充相关背景信息
            2. 使用客观、规范的表述，符合百科内容风格
            3. 包含2-3个与核心内容相关的细节描述
            4. 长度50-80字，仅返回补充材料本身，不添加任何解释说明"""
        else:  # qa
            system_prompt = """你是专业问答辅助助手，基于问题生成一段补充材料：
            1. 围绕问题核心展开，补充相关背景信息
            2. 使用客观、规范的表述，符合百科内容风格
            3. 包含2-3个与问题相关的细节描述，帮助定位答案
            4. 长度50-80字，仅返回补充材料本身，不添加任何解释说明"""
        
        prompt = f"根据以下{ '文本' if function_type == 'verify' else '问题' }生成补充材料：\n\n{text}"
        related_materials = llm.generate(prompt, system_prompt).strip()
        logger.info(f"第{generation_num+1}次补充材料生成完成，耗时: {round(time.time()-start, 4)}秒")
        return related_materials
    except Exception as e:
        logger.error(f"补充材料生成错误: {str(e)}")
        return ""

def parse_verify_result(llm_result):
    """解析校验模式下的LLM结果，仅提取修正文本内容（完全去除所有标记）"""
    # 初始化默认结果
    parsed = {
        "corrected_text": "",  # 只包含修正后的文本内容
        "explanation": ""      # 只包含修正说明内容
    }
    
    if not llm_result:
        logger.warning("LLM返回的校验结果为空")
        return parsed
    
    try:
        # 提取修正文本内容（先尝试按标记提取）
        if "修正文本：" in llm_result:
            corrected_text_match = re.search(r'修正文本：(.*?)(?=修正说明：|$)', llm_result, re.DOTALL)
            if corrected_text_match:
                parsed["corrected_text"] = corrected_text_match.group(1).strip()
        else:
            # 如果没有"修正文本："标记，尝试提取所有内容直到可能的说明部分
            end_pattern = re.compile(r'修正说明：|解释：|说明：', re.IGNORECASE)
            match = end_pattern.search(llm_result)
            if match:
                parsed["corrected_text"] = llm_result[:match.start()].strip()
            else:
                # 如果没有任何说明标记，认为整个内容都是修正文本
                parsed["corrected_text"] = llm_result.strip()
        
        # 提取修正说明内容
        if "修正说明：" in llm_result:
            explanation_match = re.search(r'修正说明：(.*)', llm_result, re.DOTALL)
            if explanation_match:
                parsed["explanation"] = explanation_match.group(1).strip()
        
        # 安全检查：如果修正文本仍包含"修正文本："前缀，则彻底清除
        parsed["corrected_text"] = re.sub(r'^修正文本：\s*', '', parsed["corrected_text"], flags=re.IGNORECASE)
        
        return parsed
    except Exception as e:
        logger.error(f"解析校验结果时出错: {str(e)}")
        return parsed

def find_closest_reference(target_text, references):
    """让模型找出与目标文本最接近的基准材料，适用于校验和问答模式"""
    if not llm or not target_text or not references:
        return None
        
    try:
        start = time.time()
        system_prompt = """你是一个专业的文本匹配专家，擅长找出与目标文本最接近的参考材料。
        请仔细对比目标文本和提供的所有参考材料，找出内容最相似、最相关的那一条。
        仅返回最匹配的参考材料的完整内容，不要添加任何解释、说明或其他内容。
        如果没有明显匹配的材料，返回"无匹配材料"。"""
        
        # 为每个参考材料添加编号，方便模型识别
        numbered_references = []
        for i, ref in enumerate(references, 1):
            ref_content = ref.get('content', '')
            ref_title = ref.get('doctitle', '无标题')
            numbered_references.append(f"参考材料{i}：{ref_title} - {ref_content}")
        
        references_text = "\n\n".join(numbered_references)
        
        prompt = f"""目标文本：{target_text}
        
        参考材料列表：
        {references_text}
        
        请从上述参考材料中找出与目标文本内容最接近、最相关的那一条，仅返回该参考材料的完整内容，不要添加任何其他信息。"""
        
        closest_ref = llm.generate(prompt, system_prompt).strip()
        logger.info(f"找到最接近的参考材料，耗时: {round(time.time()-start, 4)}秒")
        
        # 找到匹配的原始参考材料对象
        for ref in references:
            if closest_ref in ref.get('content', '') or ref.get('content', '') in closest_ref:
                return ref
                
        return {"content": closest_ref, "doctitle": "匹配结果", "note": "此为模型判断的最接近内容"}
        
    except Exception as e:
        logger.error(f"查找最接近的参考材料时出错: {str(e)}")
        return None

def find_most_relevant_content(text_to_search, search_results, function_type):
    """使用大模型从检索结果中找到与文本最相关的内容，根据功能类型调整策略"""
    if not text_to_search or not search_results or not llm:
        # 如果没有大模型，返回原始文本
        logger.info("LLM不可用，返回原始文本")
        return {
            "processed_content": text_to_search,
            "corrected_text": text_to_search,  # 仅包含文本内容
            "explanation": "",
            "closest_reference": None
        }
        
    start = time.time()
    try:
        # 根据功能类型设置不同的系统提示，统一格式要求
        if function_type == "verify":
            system_prompt = """你是一个专业的内容校验专家，擅长基于提供的所有基准材料验证并更正文本。
            请仔细分析待验证文本和所有提供的基准材料，根据所有基准材料的内容，对需要验证的文本进行全面更正。
            修正后的结果需要与对应的基准材料完全一致，包括事实、数据和表述。
            必须按照以下格式输出，不允许添加任何其他内容：
            修正文本：[在此处填写修正后的完整文本]
            修正说明：
            [错误类型1]“原文内容1”——>“修正后内容1”
            [错误类型2]“原文内容2”——>“修正后内容2”
            [错误类型3]“原文内容3”——>“修正后内容3”
            ...
            （每个修正项单独成行）"""
        else:  # qa
            system_prompt = """你是一个专业的问答专家，擅长基于所有提供的基准材料回答问题。
            请仔细分析问题和所有提供的基准材料，按以下步骤进行处理：
            1. 基于所有基准材料的综合信息提供两部分回答：
               a. 简洁回答：根据待查询问题和所有基准材料，总结出最精炼的回答（不超过30个字）
               b. 详细回答：基于所有基准材料的综合信息，提供完整、详细的回答
               两部分回答用"简洁回答："和"详细回答："作为开头明确区分，且两部分之间必须空一行
            2. 如果没有找到相关的基准材料或基准材料不足以回答问题，请直接返回"无法回答"
            
            核心要求：
            - 所有回答必须完全基于提供的所有基准材料，不得添加外部知识
            - 绝对不提及任何关于基准材料、信息来源或材料的存在
            - 不要包含任何其他标题、说明或解释性文字"""
        
        # 精简词条格式，仅保留标题和内容的组合，去除编号和格式标记
        references = "\n\n".join([
            f"{item.get('doctitle', '无标题')}：{item['content']}" 
            for item in search_results
        ])
        
        if function_type == "verify":
            prompt = f"""待验证文本：{text_to_search}
            
            基准材料：
            {references}
            
            请基于上述基准材料进行综合校验和更正，严格按照系统提示中的格式返回结果，不要包含任何额外信息。"""
        else:
            prompt = f"""待查询问题：{text_to_search}
            
            所有基准材料如下：
            {references}
            
            请基于上述基准材料进行综合回答，严格按照系统提示中的要求和格式返回结果，不要包含任何额外信息。"""
        
        llm_result = llm.generate(prompt, system_prompt)
        # 简单清洗结果
        llm_result = llm_result.strip()
        logger.info(f"内容处理完成，耗时: {round(time.time()-start, 4)}秒")
        
        # 针对校验模式进行特殊处理
        if function_type == "verify":
            parsed_result = parse_verify_result(llm_result)
            
            # 找到最相近的基准材料
            closest_reference = find_closest_reference(parsed_result["corrected_text"], search_results)
            
            return {
                "processed_content": llm_result,  # 完整结果（供日志和调试）
                "corrected_text": parsed_result["corrected_text"],  # 仅包含修正后的文本内容
                "explanation": parsed_result["explanation"],  # 单独的修正说明
                "closest_reference": closest_reference
            }
        else:
            # 问答模式处理
            closest_reference = find_closest_reference(text_to_search, search_results)
            return {
                "processed_content": llm_result,
                "corrected_text": "",  # 问答模式不需要修正文本
                "explanation": "",
                "closest_reference": closest_reference
            }
            
    except Exception as e:
        logger.error(f"内容处理时发生错误: {str(e)}", exc_info=True)
        # 发生错误时返回原始文本
        return {
            "processed_content": text_to_search,
            "corrected_text": text_to_search,  # 错误情况下仍仅返回文本内容
            "explanation": f"处理时发生错误: {str(e)}",
            "closest_reference": None
        }

def normalize_score(score, original_min, original_max):
    """将分数归一化到0-1范围"""
    if original_max == original_min:
        return 0.0
    return (score - original_min) / (original_max - original_min)

def external_api_search(text: str, method_name: str) -> dict:
    """调用外部API进行检索"""
    start = time.time()
    try:
        response = requests.post(SEARCH_API_URL, json={"text": text}, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"外部API请求失败，状态码: {response.status_code}")
            return {"items": [], "combined_content": ""}
        
        result = response.json()
        items = []
        if "result" in result and isinstance(result["result"], list):
            for item in result["result"]:
                raw_score = item.get("score", 0)
                normalized = normalize_score(raw_score, EXTERNAL_SCORE_SCALE[0], EXTERNAL_SCORE_SCALE[1])
                items.append({
                    "content": item.get("content", ""),
                    "doctitle": item.get("word", ""),
                    "score": raw_score,
                    "normalized_score": round(normalized, 4),
                    "source_method": method_name
                })
        
        combined_content = "\n\n".join([item["content"] for item in items if item["content"]])
        logger.info(f"外部API检索完成，返回{len(items)}条结果，耗时: {round(time.time()-start, 4)}秒")
        return {"items": items, "combined_content": combined_content}
        
    except Exception as e:
        logger.error(f"外部API检索错误: {str(e)}")
        return {"items": [], "combined_content": ""}

def run_combined_search(text, use_llm_enhancement, function_type):
    """生成材料并执行检索，使用空行合并文本"""
    processed_text = text
    combined_texts = [processed_text]
    
    # 记录生成的材料信息
    generation_info = {
        "processed_original_text": processed_text,
        "generated_materials": [],
        "total_generated": 0,
        "valid_generated": 0,
        "generation_errors": 0
    }
    
    # 生成相关材料
    if use_llm_enhancement and llm:
        generated_materials = []
        generation_errors = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_GENERATION) as gen_executor:
            gen_futures = [
                gen_executor.submit(generate_related_materials, processed_text, i, function_type)
                for i in range(GENERATION_COUNT)
            ]
            
            for i, future in enumerate(as_completed(gen_futures)):
                try:
                    material = future.result()
                    if material and len(material.strip()) > 10:
                        generated_materials.append(material)
                    else:
                        logger.warning(f"第{i+1}次材料生成内容无效")
                except Exception as e:
                    logger.error(f"材料生成任务失败: {str(e)}")
                    generation_errors += 1
        
        # 更新生成信息
        generation_info["generated_materials"] = generated_materials
        generation_info["total_generated"] = GENERATION_COUNT
        generation_info["valid_generated"] = len(generated_materials)
        generation_info["generation_errors"] = generation_errors
        combined_texts.extend(generated_materials)
    
    # 合并文本并执行检索 - 仅使用空行分隔
    final_query = "\n\n".join(combined_texts)
    method_name = f"combined_search_with_llm_{function_type}" if use_llm_enhancement else f"basic_combined_search_{function_type}"
    result = external_api_search(final_query, method_name)
    
    return {
        "items": result["items"],
        "combined_content": result["combined_content"],
        "generation_info": generation_info,
        "final_query": final_query
    }

@app.route('/search_references', methods=['POST'])
def search_references_endpoint():
    start = time.time()
    try:
        data = request.get_json()
        if not data or not data.get("text_to_search"):
            return jsonify({"error": "缺少 'text_to_search' 参数"}), 400
        
        # 获取请求参数
        text_to_search = data.get("text_to_search")
        use_llm_enhancement = data.get("use_llm_enhancement", False)
        function_type = data.get("function_type")
        
        # 验证文本类型参数
        if function_type not in ["verify", "qa"]:
            return jsonify({"error": "无效的 'function_type' 参数，必须是 'verify' 或 'qa'"}), 400
        
        # 检查LLM可用性
        if use_llm_enhancement and not llm:
            logger.warning("LLM未初始化，使用基础检索")
            use_llm_enhancement = False
        
        logger.info(f"检索参数 - 文本长度: {len(text_to_search)}, 大模型增强: {use_llm_enhancement}, 功能类型: {function_type}")
        
        # 执行检索
        search_result = run_combined_search(text_to_search, use_llm_enhancement, function_type)
        all_items = search_result["items"]
        generation_info = search_result["generation_info"]
        final_query = search_result["final_query"]
        
        # 处理结果
        sorted_results = sorted(all_items, key=lambda x: x["normalized_score"], reverse=True)
        deduped_results = sorted_results  # 不执行去重
        limited_results = deduped_results[:MAX_RESULTS]
        
        # 调用大模型处理内容
        processed_result = find_most_relevant_content(text_to_search, limited_results, function_type)
        
        # 准备返回结果
        combined_content = "\n\n".join([item["content"] for item in limited_results if item["content"]])
        total_duration = round(time.time() - start, 4)
        result_count = len(limited_results)
        
        # 清理返回字段
        for item in limited_results:
            item.pop("normalized_score", None)
            item.pop("source_method", None)
        
        # 构建返回数据结构
        response_data = {
            "status": "success",
            "processed_content": processed_result["processed_content"],
            "reference_material": combined_content,
            "items": limited_results,
            "processing_time_seconds": total_duration,
            "result_count": result_count,
            "use_llm_enhancement": use_llm_enhancement,
            "function_type": function_type,
            "intermediate_results": {
                "generation_info": generation_info,
                "final_combined_query": final_query,
                "deduplication_stats": {
                    "before": len(sorted_results),
                    "after": len(deduped_results),
                    "removed": 0
                }
            }
        }
        
        # 对于校验模式，添加额外的返回字段
        if function_type == "verify":
            # 修正文本仅包含处理后的文本内容，完全去除所有前缀和说明
            response_data["processed_content"] = processed_result["corrected_text"]
            # 修正说明单独返回
            response_data["verify_components"] = {
                "explanation": processed_result["explanation"]
            }
            response_data["closest_reference"] = processed_result["closest_reference"]
        else:
            # 问答模式也返回最接近的参考材料
            response_data["closest_reference"] = processed_result["closest_reference"]
        
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"API错误: {e}，耗时: {round(time.time()-start, 4)}秒", exc_info=True)
        return jsonify({"error": "检索处理失败", "details": str(e)}), 500

if __name__ == '__main__':
    logger.info("检索服务初始化成功，准备就绪")
    app.run(host='0.0.0.0', port=8081, debug=True)
