import React, { useMemo } from 'react'
import {ApiClientRest} from '../../../rest/api_client_rest'
import Chat from './Chat'

const Demo: React.FC = () => {
    const client = useMemo(() => new ApiClientRest(), [])

    return (
        <Chat client={client} />
    )
}

export default Demo
