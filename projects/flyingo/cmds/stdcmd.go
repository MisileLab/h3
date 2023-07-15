package cmds

import (
	"syscall"
	"github.com/df-mc/dragonfly/server/cmd"
)

type StopCommand struct {}

func (c StopCommand) Run(source cmd.Source, output *cmd.Output) {
	syscall.Kill(syscall.Getpid(), syscall.SIGINT)
}
