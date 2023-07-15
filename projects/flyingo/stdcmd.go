package main

import (
	"github.com/df-mc/dragonfly/server/cmd"
)

type StopCommand struct {}

func (c StopCommand) Run(source cmd.Source, output *cmd.Output) {
	source.World().Close();
}
