package main

import (
	"github.com/df-mc/dragonfly/server/cmd"
)

type NoParamCommand struct {}

func (c NoParamCommand) Run(source cmd.Source, output *cmd.Output) {
	source.World().Close();
}
