-- citation-fix.lua
-- convert wikilink-style citations to pandoc citations with page numbers

function Link(elem)
  local target = elem.target
  
  -- check if the link target starts with '@'
  if target:match("^@") then
    local cite_key = target:match("^@(.+)$")
    
    if cite_key then
      local citation = pandoc.Citation(cite_key, pandoc.NormalCitation)
      
      -- check if link content has page information
      local content_text = pandoc.utils.stringify(elem.content)
      -- match patterns like "p. 31", "p. 31~33", "p. 31 ~ 33", "p. 31-33", "p. 31 - 33"
      local page_match = content_text:match("^p%.%s*([%d~%-%s]+)$")
      
      if page_match then
        -- remove extra spaces for cleaner output
        local clean_page = page_match:gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
        citation.suffix = {pandoc.Space(), pandoc.Str(clean_page)}
      end
      
      -- create display text
      local display_text = "[@" .. cite_key
      if page_match then
        local clean_page = page_match:gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
        display_text = display_text .. ", p. " .. clean_page
      end
      display_text = display_text .. "]"
      
      local display = pandoc.Inlines({pandoc.Str(display_text)})
      return pandoc.Cite(display, {citation})
    end
  end
  
  return elem
end