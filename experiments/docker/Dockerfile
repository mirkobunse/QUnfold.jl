FROM julia:1.6

# create the user and install dependencies
ARG user
ARG uid
ARG group
ARG gid
ENV PYTHON=""
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        gettext \
        git \
        less \
        nano \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid $gid $group && \
    useradd --gid $gid --uid $uid --create-home $user

# install all Julia packages
ADD setup.jl /home/$user/.julia/config/
ADD .Project.toml /home/$user/.julia/environments/v1.6/Project.toml
ADD .Manifest.toml /home/$user/.julia/environments/v1.6/Manifest.toml
RUN chown --recursive $user:$gid /home/$user/
USER $uid
RUN julia --optimize=3 /home/$user/.julia/config/setup.jl

CMD ["bash"]
