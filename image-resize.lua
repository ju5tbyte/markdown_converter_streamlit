-- image-size.lua
-- Pandoc Lua Filter to handle Markdown-style image resizing (e.g., ![caption|width](image.jpg))

function Figure(elem)
  -- Figure의 캡션 처리
  if elem.caption and elem.caption.long and #elem.caption.long > 0 then
    local caption_text = pandoc.utils.stringify(elem.caption.long)
    
    local parts = {}
    for part in string.gmatch(caption_text, "[^|]+") do
      table.insert(parts, part)
    end
    
    if #parts > 1 then  -- 최소 2개 이상의 파트가 있어야 함 (캡션|너비)
      local last_part = parts[#parts]
      last_part = last_part:match("^%s*(.-)%s*$")
      local width = last_part:match("^(%d+)")
      
      if width then
        -- Figure 내부의 Image에 너비 설정
        if elem.content and #elem.content > 0 then
          for i, item in ipairs(elem.content) do
            if item.t == "Plain" or item.t == "Para" then
              for j, inline in ipairs(item.content) do
                if inline.t == "Image" then
                  inline.attributes['width'] = width
                end
              end
            end
          end
        end
        
        -- 캡션 업데이트: 너비 부분 제거
        table.remove(parts)
        local new_caption_str = table.concat(parts, '|')
        
        -- 빈 문자열이 아닐 때만 캡션 업데이트
        if new_caption_str ~= '' then
          -- 기존 캡션 블록 구조를 참조하여 업데이트
          local first_block = elem.caption.long[1]
          if first_block.t == "Plain" then
            elem.caption.long = { pandoc.Plain({ pandoc.Str(new_caption_str) }) }
          elseif first_block.t == "Para" then
            elem.caption.long = { pandoc.Para({ pandoc.Str(new_caption_str) }) }
          end
        else
          -- 캡션이 비어있으면 Figure의 캡션을 비움
          elem.caption.long = {}
        end
      end
    end
  end
  
  return elem
end