-- remove-hr-for-doc.lua

function HorizontalRule (elem)
  -- If format is LaTeX, PDF, or DOCX,
  -- remove the horizontal rule
  if FORMAT == "latex" or FORMAT == "pdf" or FORMAT == "docx" then
    return {}
  
  -- Otherwise, return the element unchanged
  else
    return elem
  end
end