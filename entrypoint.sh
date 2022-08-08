# An entrypoint.sh is a shell script you can execute as the command entrypoint to your
# containerized application.

# Extract Arguments (room for adding here more dynamic arguments )
for i in "$@"
do
case $i in
    -p=*|--port=*)
    PORT="${i#*=}"
    shift # past argument=value
esac
done

# Launch Jupyter in the Background
echo "[INFO]: Launching Jupyter Lab"
jupyter lab \
    --NotebookApp.token="" \
    --ip="0.0.0.0" \
    --port=${PORT} \
    --notebook-dir=/app \
    --no-browser \
    --allow-root
