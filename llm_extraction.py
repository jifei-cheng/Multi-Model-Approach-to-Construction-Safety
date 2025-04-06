import json
import pandas as pd
import os
import time
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import re

# Please enter your API keys
DEEPSEEK_API_KEY = ""  
CHATGPT_API_KEY = ""

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-reasoner"

# client = OpenAI(api_key=CHATGPT_API_KEY)
# MODEL = "gpt-3.5-turbo"


Key_Index = {
    "X1": "安全意识淡薄、自我保护能力差",
    "X2": "资质证书虚假、施工经验不足",
    "X3": "未采取安全防护措施",
    "X4": "未按照施工方案操作",
    "X5": "违规冒险作业",
    "X6": "隐患排查、勘测工作未落实",
    "X7": "岩体失稳、土方稳定性差",
    "X8": "浮土过多、推土过高",
    "X9": "土质松软",
    "X10": "混泥土路面松动",
    "X11": "基坑底部积水",
    "X12": "沟坑、沟槽边坡失稳",
    "X13": "墙体、土体抗剪强度低",
    "X14": "封堵墙接缝质量缺陷",
    "X15": "墙体砌筑工程质量不合格",
    "X16": "监管、监理、安全检查不到位",
    "X17": "安全教育培训不到位",
    "X18": "生产主体责任落实不到位",
    "X19": "技术交底不落实",
    "X20": "施工组织不到位",
    "X21": "未建立健全风险分级管控机制",
    "X22": "监督隐患排查治理工作未落实",
    "X23": "未及时办理施工许可和质检手续",
    "X24": "未及时报批施工图设计",
    "X25": "管理人员不到岗履职",
    "X26": "未建立健全应急救援管理预案",
    "X27": "抢险救援过程措施不力",
    "X28": "暴雨、雨雪",
    "X29": "寒冷冰冻",
    "X30": "风化岩体",
    "X31": "阴山坎",
    "X32": "砂质粉土",
    "X33": "流沙",
    "X34": "未预埋钢筋混泥土",
    "X35": "未加强钢筋或圈梁",
    "X36": "沟槽开挖不规范",
    "X37": "挖掘机超载施工",
    "X38": "对边坡、土层进行扰动",
    "X39": "反复碾压和巨大震动",
    "X40": "放坡措施不到位",
    "X41": "支护、固壁措施不到位",
    "X42": "未做好防水浸泡措施",
    "X43": "未设置隔水排水措施",
    "X44": "边坡防护不当",
    "X45": "补救措施不当"
}

def load_excel_data(file_path, column_name):
    df = pd.read_excel(file_path)
    if column_name not in df.columns:
        return None
    return df

def chat(text):
    user_prompt = f"""
任务角色：
    你是一位安全事故分析专家，专注于从事故文本中识别并抽取已定义的事故因素。

任务目标：
    请根据以下定义的事故因素集合 {Key_Index}，从给定的事故文本中抽取与之相关的因素编号，仅返回一个列表。

输出格式：
    输出示例格式如下（仅包含编号，无需解释）：X1, X5, X12, X22

待处理文本：
    （一）直接原因 现场作业人员将用于基坑支护的拉森钢板桩违规拔出，造成基坑事发时无支护措施；基坑底部积水，未及时抽排，泡涨基坑侧壁导致基坑边坡自稳能力下降。此外，基坑边坡侧，违规堆放大量浮土和挖掘机超载施工也是造成此次坍塌事故的重要原因。(现场示意图见附件) （二）管理原因 1.武汉鑫润祥物资有限公司：一是违法承揽工程，借用武汉市市政建设综合开发公司资质参与投标；二是将工程承包给不具备相关资质的个人。 2.武汉市市政建设综合开发公司：向武汉鑫润祥物资有限公司转借营业执照和资质证书等资料，以本公司名义承揽工程，并在招投标环节为其弄虚作假、骗取中标提供便利。 3.上海容基工程管理有限公司：一是资格审查不到位，未核实施工现场人员身份和资格；二是现场安全监理责任不落实，在基坑施工时未安排旁站监理，未及时发现和制止施工人员擅自拆除基坑防护体系的违章作业行为；三是专项方案审查不到位，未督促施工单位编制基坑作业专项施工方案。 4.武汉市东西湖城市建设投资发展有限公司：一是未办理项目施工许可和质量安全监督手续；二是未及时向图审部门报批施工图设计。 5.东西湖区水务和湖泊局：一是对事发工程合规手续办理、现场施工等环节监督、指导不到位，对参建各方的违法行为失察；二是落实“迎大庆、保军运”战时期间安全生产隐患排查和专项整治要求不到位。
返回：X2, X3, X5, X7, X8, X11, X16, X21, X22, X23, X24, X37, X41

待处理文本内容：{text}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是一位专注于安全管理和事故分析的专家，擅长从事故案例中识别和抽取关键致因因素。"},
            {"role": "user", "content": user_prompt}
        ],
    )
    if response.choices[0].message.content:
        return response.choices[0].message.content
    else:
        print(f"Error in model response: {response}")
        return None

def process_text(row, column_name):
    """
    Receives a full row of data, analyzes the specified column, adds a label, and returns the full row
    """
    text = row[column_name]
    result = chat(text)
    if result:
        try:
            cleaned_text = re.sub(r'^"""\s*|\s*"""$', '', result, flags=re.DOTALL)
            cleaned_text = re.sub(r'^\s*```json|```$', '', cleaned_text.strip())
            row["factor"] = cleaned_text.strip()
            return row
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    return None

def save_to_excel(results, filename="output.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Data successfully saved to {filename}")

def main():
    input_file = "data.xlsx"
    column_name = "reason"
    output_file = "new-ds.xlsx"
    batch_size = 1
    max_workers = 1  # Safer thread count

    data = load_excel_data(input_file, column_name)
    if data is None:
        print(f"Column {column_name} not found")
        return

    results = []
    processed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_text, row.copy(), column_name) for _, row in data.iterrows()]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing progress"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    processed_count += 1

                    if processed_count % batch_size == 0:
                        print(f"\nProcessed {processed_count} records")
            except Exception as e:
                print(f"Error during task processing: {e}")

    if results:
        save_to_excel(results, output_file)
        print(f"\nFinally saved {len(results)} records")

    print("Processing complete")

if __name__ == "__main__":
    main()