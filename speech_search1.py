import os
import json
import logging
import time
import sys
import requests
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import random
from pathlib import Path

from src.remove import remove_duplicates
# 配置和初始化
load_dotenv()
app = Flask(__name__)
CORS(app)  # 允许跨域请求，与前端配合

# 检索配置
SEARCH_API_URL = "http://222.29.51.95:4042/speech_search"
MAX_WORKERS_GENERATION = 3  # 生成材料的线程数
GENERATION_COUNT = 3  # 大模型生成材料的次数
MAX_RESULTS = 15  # 限制返回的最大词条数量

# ARK API配置
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL_ID = "deepseek-v3-1-250821"
EXTERNAL_SCORE_SCALE = (0, 100)  # 外部API分数范围

# 日志配置
logging.basicConfig(
    filename="search_api.log",
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
        llm = ArkApiAdapter(
            api_key=ARK_API_KEY, 
            base_url=ARK_BASE_URL, 
            model=ARK_MODEL_ID
        )
        logger.info("LLM客户端初始化成功")
    else:
        logger.warning("未配置ARK_API_KEY，大模型增强功能将不可用")
except Exception as e:
    logger.error(f"LLM客户端初始化失败: {str(e)}", exc_info=True)

def generate_related_materials(text: str, generation_num: int, function_type: str) -> str:
    """使用LLM生成与文本相关的材料"""
    if not text or not llm:
        return ""
        
    start = time.time()
    try:
        # 根据功能类型设置不同的系统提示
        if function_type == "verify":
            system_prompt = """你是专业内容验证专家，生成辅助验证的补充材料：
            1. 包含原文本所有核心关键词
            2. 围绕原文本主题提供验证所需关键信息
            3. 不引入无关内容，长度约50字
            4. 只返回补充材料本身，不添加任何解释"""
        else:  # qa
            system_prompt = """你是专业问答辅助专家，生成辅助问答的补充材料：
            1. 包含问题中所有核心关键词
            2. 围绕问题展开，提供答案方向和背景信息
            3. 不引入无关内容，长度约50字
            4. 只返回补充材料本身，不添加任何解释"""
        
        prompt = f"根据以下{ '文本' if function_type == 'verify' else '问题' }生成补充材料：\n\n{text}"
        related_materials = llm.generate(prompt, system_prompt).strip()
        logger.info(f"第{generation_num+1}次材料生成完成，耗时: {round(time.time()-start, 4)}秒")
        return related_materials
    except Exception as e:
        logger.error(f"材料生成错误: {str(e)}")
        return ""

def find_most_relevant_content(text_to_search, search_results, function_type):
    """使用大模型从检索结果中找到与文本最相关的内容，根据功能类型调整策略"""
    logger.info(text_to_search)
    logger.info(search_results)
    logger.info(function_type)
    
    if not text_to_search or not search_results or not llm:
        # 如果没有大模型，返回原始文本
        return text_to_search
        
    start = time.time()
    try:
        # 根据功能类型设置不同的系统提示
        if function_type == "verify":
            system_prompt = """你是一个专业的内容校验专家，擅长基于提供的基准材料验证文本。
            请仔细分析待验证文本和所有提供的基准材料，按以下步骤进行处理：
            1. 首先从所有基准材料中挑选出与待验证文本内容最相似的一个
            2. 对比待验证文本与挑选出的最相似基准材料，找出所有不同之处
            3. 按照以下格式返回结果：
               a. 修正文本：直接使用你挑选出的最相似的基准材料内容
               b. 修正说明：详细列出待验证文本与基准材料的所有不同之处，并指明待验证文本中的错误
               两部分结果用"修正文本："和"修正说明："作为开头明确区分
            
            核心要求：
            - 所有判断必须完全基于提供的基准材料，不得添加外部知识
            - 确保准确识别最相似的基准材料，即使存在多个相关材料
            - 修正说明中要清晰指出差异点和错误所在
            - 不要包含任何其他标题、说明或解释性文字（每个修正项单独成行）"""
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
        
        # 精简词条格式，仅保留标题和内容的组合
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
        
        relevant_content = llm.generate(prompt, system_prompt)
        logger.info(relevant_content)
        # 简单清洗结果
        relevant_content = relevant_content.strip()
        logger.info(f"内容处理完成，耗时: {round(time.time()-start, 4)}秒")
        logger.info(relevant_content)
        
        return relevant_content
    except Exception as e:
        logger.error(f"内容处理时发生错误: {str(e)}", exc_info=True)
        # 发生错误时返回原始文本
        return text_to_search

def extract_verify_components(llm_response):
    """从大模型响应中提取修正文本和修正说明"""
    try:
        # 提取修正文本
        corrected_text_match = re.search(r'修正文本：(.*?)(?=修正说明：|$)', llm_response, re.DOTALL)
        corrected_text = corrected_text_match.group(1).strip() if corrected_text_match else ""
        
        # 提取修正说明
        explanation_match = re.search(r'修正说明：(.*?)$', llm_response, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        return {
            "corrected_text": corrected_text,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"提取校验组件失败: {str(e)}")
        return {
            "corrected_text": llm_response,
            "explanation": "未能解析修正说明"
        }

def extract_qa_components(llm_response):
    """从大模型响应中提取问答的简洁回答和详细回答"""
    try:
        # 提取简洁回答
        brief_match = re.search(r'简洁回答：(.*?)(?=详细回答：|$)', llm_response, re.DOTALL)
        brief_answer = brief_match.group(1).strip() if brief_match else ""
        
        # 提取详细回答
        detailed_match = re.search(r'详细回答：(.*?)$', llm_response, re.DOTALL)
        detailed_answer = detailed_match.group(1).strip() if detailed_match else ""
        
        # 处理无法回答的情况
        if "无法回答" in llm_response:
            return {
                "brief_answer": "无法回答",
                "detailed_answer": "没有找到相关的基准材料或基准材料不足以回答问题"
            }
            
        return {
            "brief_answer": brief_answer,
            "detailed_answer": detailed_answer
        }
    except Exception as e:
        logger.error(f"提取问答组件失败: {str(e)}")
        return {
            "brief_answer": llm_response[:100],  # 截断长文本
            "detailed_answer": llm_response
        }

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

def normalize_score(score, original_min, original_max):
    """将分数归一化到0-1范围"""
    if original_max == original_min:
        return 0.0
    return (score - original_min) / (original_max - original_min)

def external_api_search(text: str, method_name: str, function_type: str, start_time: int = 0, end_time: int = 0) -> dict:
    """调用外部API进行检索，处理时间参数和文本清理，新增function_type参数"""
    start = time.time()
    try:
        # 发送请求，添加type参数
        payload = {
            "text": text, 
            "start_time": start_time, 
            "end_time": end_time,
            "type": function_type  # 新增type参数，值为"verify"或"qa"
        }
        logger.debug(f"API请求: {SEARCH_API_URL}, {payload}")
        response = requests.post(
            SEARCH_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
            },
            timeout=300
        )

        # 响应处理
        if response.status_code != 200:
            msg = f"API请求失败: {response.status_code}"
            if response.status_code == 500:
                msg += f", 响应: {response.text[:500]}"
            logger.error(msg)
            return {"items": [], "combined_content": ""}

        # 解析结果
        result = response.json()
        items = []
        if "result" in result and isinstance(result["result"], list):
            for item in result["result"]:
                raw_score = item.get("score", 0)
                items.append({
                    "content": item.get("content", ""),
                    "doctitle": item.get("doctitle", ""),
                    "location": item.get("location", ""),
                    "score": raw_score,
                    "normalized_score": round((raw_score - EXTERNAL_SCORE_SCALE[0]) / 
                                             (EXTERNAL_SCORE_SCALE[1] - EXTERNAL_SCORE_SCALE[0]) 
                                             if EXTERNAL_SCORE_SCALE[1] != EXTERNAL_SCORE_SCALE[0] else 0, 4),
                    "source": item.get("source", ""),
                    "speechtime": item.get("speechtime", 0),
                    "pubdate": item.get("pubdate", ""),
                    "source_method": method_name
                })

        combined = "\n\n".join([i["content"] for i in items if i["content"]])
        logger.info(f"API检索完成，{len(items)}条结果，耗时{round(time.time()-start, 4)}秒")
        return {"items": items, "combined_content": combined}
        
    except Exception as e:
        logger.error(f"API检索错误: {str(e)}", exc_info=True)
        return {"items": [], "combined_content": ""}

def run_combined_search(text, use_llm_enhancement, function_type, start_time: int = None, end_time: int = None):
    """生成材料并执行检索，新增时间范围参数"""
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
    
    # 合并文本并执行检索 - 仅使用空行分隔文本
    final_query = "\n\n".join(combined_texts)
    method_name = f"combined_search_with_llm_{function_type}" if use_llm_enhancement else f"basic_combined_search_{function_type}"
    # 传递时间参数和类型参数到API调用
    result = external_api_search(final_query, method_name, function_type, start_time, end_time)
    
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
        
        # 解析参数
        text = data.get("text_to_search")
        use_llm = data.get("use_llm_enhancement", False)
        func_type = data.get("function_type")
        
        # 验证文本类型参数
        if func_type not in ["verify", "qa"]:
            return jsonify({"error": "无效的 'function_type' 参数，必须是 'verify' 或 'qa'"}), 400
        
        start_time, end_time = data.get("start_time"), data.get("end_time")

        # 时间参数验证
        try:
            if start_time is None:
                start_time = 0
            if end_time is None:
                end_time = 0
            if start_time > end_time:
                start_time = end_time = 0
        except (ValueError, TypeError):
            start_time = end_time = 0  
        
        # LLM可用性检查
        if use_llm and not llm:
            logger.warning("LLM不可用，使用基础检索")
            use_llm = False

        # 执行检索
        search_result = run_combined_search(text, use_llm, func_type, start_time, end_time)
        items = search_result["items"]
        gen_info = search_result["generation_info"]

        # 处理结果
        sorted_items = sorted(items, key=lambda x: x["normalized_score"], reverse=True)
        deduped = remove_duplicates(sorted_items)[:MAX_RESULTS]
        processed_content = find_most_relevant_content(text, deduped, func_type)
        combined_content = "\n\n".join([i["content"] for i in deduped if i["content"]])
        
        # 清理返回字段
        for item in deduped:
            item.pop("normalized_score", None)
            item.pop("source_method", None)
        
        # 初始化组件和最接近的参考材料变量
        components = None
        closest_reference = None
        target_text = processed_content  # 默认使用处理后的内容作为匹配目标
        
        if llm:
            # 根据功能类型提取相应组件
            if func_type == "verify":
                components = extract_verify_components(processed_content)
                # 使用修正文本作为匹配目标
                target_text = components.get("corrected_text", text)
            else:  # qa
                components = extract_qa_components(processed_content)
                # 使用详细回答作为匹配目标
                target_text = components.get("detailed_answer", processed_content)
            
            # 为两种类型都查找最接近的基准材料
            closest_reference = find_closest_reference(target_text, deduped)
        
        # 准备返回数据
        response_data = {
            "status": "success",
            "reference_material": combined_content,
            "items": deduped,
            "processing_time_seconds": round(time.time() - start, 4),
            "result_count": len(deduped),
            "use_llm_enhancement": use_llm,
            "function_type": func_type,
            "time_range": {"start_time": start_time, "end_time": end_time},
            "intermediate_results": {
                "generation_info": gen_info
            },
            # 对于校验模式，返回修正文本作为主要处理结果
            "processed_content": components.get("corrected_text", processed_content) if func_type == "verify" else processed_content,
            # 确保修正说明在顶层返回，方便前端直接使用
            "correction_explanation": components.get("explanation", "") if func_type == "verify" else "",
            "closest_reference": closest_reference
        }
        
        # 根据功能类型添加相应的组件
        if func_type == "verify":
            response_data["verify_components"] = components or {
                "corrected_text": processed_content,
                "explanation": "未能提取修正说明"
            }
        else:
            response_data["qa_components"] = components or {
                "brief_answer": processed_content[:100],
                "detailed_answer": processed_content
            }
        
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"API错误: {e}，耗时{round(time.time()-start, 4)}秒", exc_info=True)
        return jsonify({"error": "检索处理失败", "details": str(e)}), 500

if __name__ == '__main__':
    logger.info("检索服务初始化成功，准备就绪")
    app.run(host='0.0.0.0', port=8085, debug=True)
