#!/bin/bash

# Find the installation path of the verl library
verl_path=$(python -c "import verl; print(verl.__path__[0])" 2>/dev/null)

# Check if the verl library was found
if [ -z "$verl_path" ]; then
    echo "Error: verl library not found. Please ensure verl is installed correctly."
    exit 1
fi

# Construct the path to the __init__.py file
init_file="${verl_path}/__init__.py"

# Check if the __init__.py file exists
if [ ! -f "$init_file" ]; then
    echo "Error: ${init_file} not found."
    exit 1
fi

# Define the content to append
content=$'if is_npu_available():\n    from mindspeed_rl.boost import verl'

# Check if the content already exists in the file
if grep -qxF "$content" "$init_file"; then
    echo "The specified content already exists in ${init_file}, no need to append."
    exit 0
fi

# Append the content to the end of the file
echo -e "\n$content" >> "$init_file"

echo "Successfully appended the following content to ${init_file}:"
echo "$content"
