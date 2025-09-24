    
def remove_duplicates(items):
    """移除重复或包含关系的词条并智能合并标题"""
    items_copy = [item.copy() for item in items]
    to_remove = set()
    title_merges = {}
    
    for i, item_a in enumerate(items_copy):
        if i in to_remove:
            continue
            
        a_content = item_a.get('content', '').strip().replace(" ", "")
        a_title = item_a.get('doctitle', '').strip()
        
        for j, item_b in enumerate(items_copy):
            if i == j or j in to_remove:
                continue
                
            b_content = item_b.get('content', '').strip().replace(" ", "")
            b_title = item_b.get('doctitle', '').strip()
            
            # 检查内容包含关系
            if a_content and b_content:
                # A被B包含
                if a_content in b_content:
                    to_remove.add(i)
                    # 处理标题合并
                    if a_title and a_title != b_title and not (a_title.replace(" ", "").lower() in b_title.replace(" ", "").lower()):
                        title_merges.setdefault(j, {b_title}).add(a_title)
                    break
                # B被A包含
                elif b_content in a_content:
                    to_remove.add(j)
                    # 处理标题合并
                    if b_title and a_title != b_title and not (b_title.replace(" ", "").lower() in a_title.replace(" ", "").lower()):
                        title_merges.setdefault(i, {a_title}).add(b_title)
    
    # 应用标题合并并返回结果
    return [
        {**item, 'doctitle': "<br>".join(title_merges[idx])} 
        if idx in title_merges else item 
        for idx, item in enumerate(items_copy) 
        if idx not in to_remove
    ]